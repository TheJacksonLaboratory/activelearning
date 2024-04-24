import os
import napari
import zarr
import zarrdataset as zds
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import cv2
import numpy as np
from itertools import repeat
from functools import partial, reduce
import operator


class SuperPixelGenerator(zds.MaskGenerator):
    """Gerates a labeled mask based on the super pixels computed from the input
    image. The super pixels are computed using the SEEDS method.

    The axes of the input image are expected to be YXC, or YX if the image has
    no channels.
    """
    def __init__(self, iterations: int = 5, prior: int = 3,
                 num_superpixels: int = 512,
                 num_levels: int = 10,
                 histogram_bins: int = 10,
                 double_step: bool = False):

        super(SuperPixelGenerator, self).__init__(axes="YXC")
        self._iterations = iterations
        self._prior = prior
        self._num_superpixels = num_superpixels
        self._num_levels = num_levels
        self._histogram_bins = histogram_bins
        self._double_step = double_step

    def _compute_transform(self, image):
        if image.ndim > 2:
            image_channels = image.shape[2]
        else:
            image_channels = 1

        super_pixels = cv2.ximgproc.createSuperpixelSEEDS(
            image.shape[1], image.shape[0], image_channels=image_channels,
            prior=self._prior,
            num_superpixels=self._num_superpixels,
            num_levels=self._num_levels,
            histogram_bins=self._histogram_bins,
            double_step=self._double_step
        )

        norm_image = ((np.copy(image) - image.min(axis=(0, 1), keepdims=True))
                      / (image.max(axis=(0, 1), keepdims=True)
                         - image.min(axis=(0, 1), keepdims=True)))
        norm_image = 255.0 * norm_image
        norm_image = norm_image.astype(np.uint8)

        super_pixels.iterate(norm_image, self._iterations)
        labels = super_pixels.getLabels()

        if image_channels > 1:
            labels = np.expand_dims(labels, self.axes.index("C"))

        return labels


def save_mask(mask, filename, output_dir, mask_data_group="mask"):
    os.makedirs(output_dir, exist_ok=True)

    basename = ".".join(os.path.basename(filename).split(".")[:-1])
    mask_filename = os.path.join(output_dir, basename + ".zarr")

    root_grp = zarr.open(mask_filename, mode="a")

    root_grp.create_dataset(
        name=mask_data_group + "/0",
        data=mask > 0,
        dtype=bool,
        chunks=True,
        compressor=zarr.Blosc(clevel=9),
        write_empty_chunks=False,
        overwrite=True
    )

    return mask_filename


def retrieve_pyramid_loaders(filename, data_group=None, **kwargs):
    if data_group is None or data_group.lower() == "none":
        pyramid_group = [None]
    else:
        parent_group = data_group.split("/")
        if len(parent_group) > 1:
            parent_group = "/".join(parent_group[:-1] + ["%i"])
            scale_start = 0
        else:
            parent_group = "%i"
            scale_start = int(data_group)

        pyramid_group = [parent_group % s for s in range(scale_start, scale_start + 6)]

    pyramid_images = []
    pyramid_stores = []
    for grp in pyramid_group:
        try:
            image, store = zds.image2array(filename, data_group=grp)
            pyramid_images.append(image)
            pyramid_stores.append(store)

        except (KeyError, IndexError):
            break

    return pyramid_images, pyramid_stores


def generate_pyramid(pyramid_images, source_axes, axes=None, roi=None, **kwargs):
    pyramid_dask_images = []

    if axes is None:
        axes = source_axes

    permutation_order = zds.map_axes_order(source_axes, axes)

    non_required_axes = set(source_axes).difference(set(axes))

    for image in pyramid_images:
        da_image = da.from_zarr(image)[roi]
        da_image = da.transpose(da_image, permutation_order)
        da_image = da.squeeze(da_image, tuple(
            permutation_order.index(source_axes.index(ax))
            for ax in non_required_axes
        ))
        pyramid_dask_images.append(da_image)

    return pyramid_dask_images


def annotate_mask(layers_metadata, patch_size=256, **kwargs):
    layers_masks = {}
    layers_pyramid_stores = []
    active_axes = None

    viewer = napari.Viewer(title="Phase I: Masking")
    for layer_name, (image_metadata, annotate_this) in layers_metadata.items():
        pyramid_images, pyramid_stores = retrieve_pyramid_loaders(**image_metadata)
        layers_pyramid_stores += pyramid_stores

        image_pyramid = generate_pyramid(pyramid_images, **image_metadata)
        channel_axis = image_metadata["axes"].index("C") if "C" in image_metadata["axes"] else None
        is_rgb = image_pyramid[0].shape[channel_axis] == 3
        current_layer = viewer.add_image(image_pyramid, name=layer_name, multiscale=True, channel_axis=channel_axis if not is_rgb else None, rgb=is_rgb)

        if isinstance(current_layer, list):
            current_layer = current_layer[0]

        current_active_axes = list(image_metadata["axes"])
        if channel_axis is not None:
            current_active_axes.remove("C")

        if active_axes is None:
            active_axes = current_active_axes

        if annotate_this:
            mask_shape = [
                round(s / patch_size)
                for s, ax in zip(current_layer.data.shape, current_active_axes)
                if ax in "YX"
            ]

            mask_scale = [patch_size for ax in current_active_axes]

            layers_masks[layer_name] = np.zeros(mask_shape, dtype=np.int32)

            viewer.add_labels(data=layers_masks[layer_name], scale=mask_scale, name=layer_name + "-annotation", translate=[(patch_size - 1) / 2 for ms in mask_shape])

    viewer.dims.axis_labels = tuple(active_axes)

    napari.run()

    # shapes_mask = shapes_layer.to_labels(mask_shape=mask.shape)
    # if shapes_mask.size:
    #     mask = np.bitwise_or(mask > 0, shapes_mask.astype(bool))

    for store in layers_pyramid_stores:
        if store is not None:
            store.close()

    return layers_masks


def get_zarrdataset(image_metadata, mask_metadata=None, patch_size=512,
                    output_filename=None,
                    shuffle=True,
                    **superpixel_kwargs):
    data_specs = [
        zds.ImagesDatasetSpecs(
            filenames=image_metadata["filename"],
            data_group=image_metadata["data_group"],
            source_axes=image_metadata["source_axes"],
            axes=image_metadata["axes"],
            roi=image_metadata["roi"],
        )
    ]

    if output_filename is None:
        data_specs.append(
            zds.ImagesDatasetSpecs(
                filenames=image_metadata["filename"],
                data_group=image_metadata["data_group"],
                source_axes=image_metadata["source_axes"],
                axes=image_metadata["axes"],
                roi=image_metadata["roi"],
                image_loader_func=SuperPixelGenerator(**superpixel_kwargs),
                modality="superpixels"
            )
        )

    else:
        if isinstance(image_metadata["roi"], str):
            base_roi_list = zds.parse_rois(image_metadata["roi"])

        else:
            base_roi_list = image_metadata["roi"]

        if not isinstance(base_roi_list, list):
            base_roi_list = [base_roi_list]

        sp_roi = [
            tuple(base_roi[image_metadata["source_axes"].index(ax)] for ax in "YX")
            for base_roi in base_roi_list
        ]

        data_specs.append(
            zds.ImagesDatasetSpecs(
                filenames=output_filename,
                data_group="confidence_map/0",
                source_axes="YX",
                modality="superpixels",
                roi=sp_roi
            )
        )

    if mask_metadata is not None:
        data_specs.append(zds.MasksDatasetSpecs(**mask_metadata))

    train_dataset = zds.ZarrDataset(
        data_specs,
        return_positions=True,
        draw_same_chunk=True,
        patch_sampler=zds.PatchSampler(patch_size=patch_size,
                                       spatial_axes="YX"),
        shuffle=shuffle
    )

    return train_dataset


def downsample_image(output_filename, group, num_scales=5):
    for s in range(num_scales):
        source_arr = da.from_zarr(output_filename,
                                  component="%s/%i" % (group, s))
        target_arr = source_arr.map_blocks(
            cv2.resize,
            dsize=None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_NEAREST,
            chunks=tuple(tuple(round(chk / 2) for chk in chk_ax)
                         for chk_ax in source_arr.chunks),
            dtype=source_arr.dtype,
            meta=np.empty((0, ), dtype=source_arr.dtype)
        )

        with ProgressBar():
            target_arr.to_zarr(output_filename,
                               component="%s/%i" % (group, s + 1),
                               compressor=zarr.Blosc(clevel=9),
                               write_empty_chunks=False,
                               overwrite=True)


def parse_dataset_metadata(filenames):
    if isinstance(filenames, list):
        if len(filenames) == 1:
            filenames = filenames[0]

    if isinstance(filenames, str):
        if filenames.lower().endswith(".txt"):
            with open(filenames, "r") as fp:
                filenames = fp.readlines()
        else:
            filenames = [filenames]

    assert isinstance(filenames, list) and all(map(isinstance, filenames, repeat(str))), "Type of input filenames is not supported"
    dataset_metadata = list(map(partial(zds.parse_metadata, default_source_axes="TCZYX"), filenames))

    dataset_metadata = reduce(lambda l1, l2: l1 + l2, dataset_metadata)

    return dataset_metadata


def prepare_output_arrays(filename, source_axes, data_group, roi, output_dir=None, patch_size=256, **kwargs):
    output_basename = ".".join(os.path.basename(filename).split(".")[:-1])
    output_basename += ".zarr"
    output_filename = os.path.join(output_dir, output_basename)

    img_collection = zds.ImageCollection(
        dict(
            images=dict(
                filename=filename,
                source_axes=source_axes,
                data_group=data_group,
                roi=roi
            ),
        ),
    )

    image_shape = img_collection.collection["images"].shape

    confidence_shape = [
        image_shape[source_axes.index(ax)]
        for ax in "YX"
        if ax in source_axes
    ]

    confidence_chunks = [patch_size, patch_size]

    out_grp = zarr.open(output_filename)

    out_grp.create_dataset(
        name="confidence_map/0",
        shape=confidence_shape,
        chunks=confidence_chunks,
        compressor=zarr.Blosc(clevel=9),
        dtype=np.float32,
        write_empty_chunks=False,
        overwrite=True
    )

    out_grp.create_dataset(
        name="annotations/0",
        shape=confidence_shape,
        chunks=confidence_chunks,
        compressor=zarr.Blosc(clevel=9),
        dtype=np.int64,
        write_empty_chunks=False,
        overwrite=True
    )

    return output_filename


def prepare_output_dataset(output_dir, images_metadata, patch_size=256):
    output_filenames = []
    for im_metadata in images_metadata:
        output_filename = prepare_output_arrays(output_dir=output_dir, patch_size=patch_size, **im_metadata)
        output_filenames.append(output_filename)

    return output_filenames
