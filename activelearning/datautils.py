import os
import napari
import zarr
import zarrdataset as zds
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import cv2
import numpy as np
from skimage.util.shape import view_as_blocks
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


def retrieve_pyramid_loaders(filenames, data_group=None, **kwargs):
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
            image, store = zds.image2array(filenames, data_group=grp)
            pyramid_images.append(image)
            pyramid_stores.append(store)

        except (KeyError, IndexError):
            break

    return pyramid_images, pyramid_stores


def generate_pyramid(pyramid_images, roi=None, **kwargs):
    pyramid_dask_images = []
    for image in pyramid_images:
        da_image = da.from_zarr(image)[roi]
        pyramid_dask_images.append(da_image)

    return pyramid_dask_images


def annotate_mask(output_dir, layers_metadata, layer_to_annotate=None,
                  annotation_name="masks",
                  patch_size=256,
                  scale=1.0,
                  mask_dtype=bool,
                  mask_modality="masks",
                  **kwargs):
    layers_pyramid_stores = []
    active_axes = None

    mask_metadata = None
    mask_temp = None
    mask_org = None
    main_layer_shape_dict = None

    viewer = napari.Viewer(title="Phase I: Masking")
    for layer_name, image_metadata in layers_metadata.items():
        pyramid_images, pyramid_stores = retrieve_pyramid_loaders(
            **image_metadata
        )
        layers_pyramid_stores += pyramid_stores

        image_pyramid = generate_pyramid(pyramid_images, **image_metadata)
        if "C" in image_metadata["source_axes"]:
            channel_axis = image_metadata["source_axes"].index("C")
            # is_rgb = image_pyramid[0].shape[channel_axis] == 3
        else:
            channel_axis = None
            # is_rgb = False

        current_layer_shape_dict = {
            ax: s
            for s, ax in zip(image_pyramid[0].shape,
                             image_metadata["source_axes"])
        }
        if "C" in current_layer_shape_dict:
            is_rgb = current_layer_shape_dict.pop("C") == 3
        else:
            is_rgb = False

        if layer_name == "images":
            main_layer_shape_dict = current_layer_shape_dict
            current_layer_scale = [1] * len(main_layer_shape_dict)
            current_layer_translate = [0] * len(main_layer_shape_dict)
        else:
            current_layer_scale = [
                main_layer_shape_dict.get(ax, 1) / s
                for ax, s in current_layer_shape_dict.items()
            ]

            current_layer_translate = [
                (scl - 1) / 2
                for scl in current_layer_scale
            ]

        current_layer = viewer.add_image(
            image_pyramid,
            name=layer_name,
            multiscale=True,
            channel_axis=channel_axis,
            colormap=["red", "green", "blue"],
            scale=current_layer_scale,
            translate=current_layer_translate
        )

        if isinstance(current_layer, list):
            current_layer = current_layer[0]

        current_active_axes = list(image_metadata["axes"])
        if channel_axis is not None:
            current_active_axes.remove("C")

        if active_axes is None:
            active_axes = current_active_axes

        if layer_to_annotate is not None and layer_name == layer_to_annotate:
            mask_metadata = prepare_output_zarr(
                annotation_name + "/0",
                output_dir=output_dir,
                patch_size=patch_size,
                scale=scale,
                output_axes=current_active_axes,
                output_dtype=mask_dtype,
                output_modality=mask_modality,
                **image_metadata
            )

            mask_org = zarr.open(
                mask_metadata["filenames"] + "/"
                + mask_metadata["data_group"],
                mode="a")

            mask_temp = np.zeros_like(mask_org, dtype=np.int32)

            mask_scale = [1 / scale] * len(current_active_axes)

            viewer.add_labels(data=mask_temp, scale=mask_scale,
                              name=annotation_name,
                              translate=[(scl - 1) / 2 for scl in mask_scale])

    viewer.dims.axis_labels = tuple(active_axes)

    napari.run()

    # shapes_mask = shapes_layer.to_labels(mask_shape=mask.shape)
    # if shapes_mask.size:
    #     mask = np.bitwise_or(mask > 0, shapes_mask.astype(bool))

    for store in layers_pyramid_stores:
        if store is not None:
            store.close()

    if mask_metadata is not None:
        layers_metadata[annotation_name] = mask_metadata
        if type(mask_org) is bool:
            mask_org[:] = mask_temp[:] > 0
        else:
            mask_org[:] = mask_temp[:]


def get_zarrdataset(dataset_metadata, patch_size=512, shuffle=True,
                    **superpixel_kwargs):

    dataset_metadata["superpixels"] = zds.ImagesDatasetSpecs(
        filenames=dataset_metadata["images"]["filenames"],
        data_group=dataset_metadata["images"]["data_group"],
        source_axes=dataset_metadata["images"]["source_axes"],
        axes=dataset_metadata["images"]["axes"],
        roi=dataset_metadata["images"]["roi"],
        image_loader_func=SuperPixelGenerator(**superpixel_kwargs),
        modality="superpixels"
    )

    train_dataset = zds.ZarrDataset(
        list(dataset_metadata.values()),
        return_positions=True,
        draw_same_chunk=True,
        patch_sampler=zds.PatchSampler(patch_size=patch_size,
                                       spatial_axes="YX"),
        shuffle=shuffle
    )

    return train_dataset


def downsample_chunk(chunk):
    pad = [(0, s % 2) for s in chunk.shape]
    padded_chunk = np.pad(chunk, pad_width=pad)
    chunk_blocks = view_as_blocks(padded_chunk, block_shape=(2, 2))
    flatten_view = chunk_blocks.reshape(chunk_blocks.shape[0],
                                        chunk_blocks.shape[1], -1)
    max_blocks = np.max(flatten_view, axis=2)

    return max_blocks


def downsample_image(filenames, source_axes, data_group=None, num_scales=5,
                     **kwargs):
    if data_group is not None:
        root_group = "/".join(data_group.split("/")[:-1]) + "/%i"
    else:
        root_group = "%i"

    source_arr = da.from_zarr(filenames, component=data_group)
    min_size = min(source_arr.shape[source_axes.index(ax)]
                   for ax in "YX" if ax in source_axes
                   )
    num_scales = min(num_scales, int(np.log2(min_size)))

    for s in range(1, num_scales):
        target_arr = source_arr.map_blocks(
            downsample_chunk,
            chunks=tuple(tuple(np.ceil(chk / 2) for chk in chk_ax)
                         for chk_ax in source_arr.chunks),
            dtype=source_arr.dtype,
            meta=np.empty((0, ), dtype=source_arr.dtype)
        )

        with ProgressBar():
            target_arr.to_zarr(filenames,
                               component=root_group % s,
                               compressor=zarr.Blosc(clevel=9),
                               write_empty_chunks=False,
                               overwrite=True)

        source_arr = da.from_zarr(filenames,
                                  component=root_group % s)


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
    dataset_metadata = map(partial(zds.parse_metadata, default_source_axes="TCZYX"), filenames)
    dataset_metadata = reduce(lambda l1, l2: l1 + l2, dataset_metadata)
    for metadata in dataset_metadata:
        metadata["filenames"] = metadata.pop("filename")

    dataset_metadata = list(map(lambda metadata:
                                zds.ImagesDatasetSpecs(**metadata),
                                dataset_metadata))
    return dataset_metadata


def prepare_output_zarr(output_array_name, output_dir, filenames, source_axes,
                        data_group=None,
                        roi=None,
                        axes=None,
                        patch_size=256,
                        scale=1.0,
                        output_axes=None,
                        output_dtype=None,
                        output_modality=None,
                        data=None,
                        **kwargs):
    output_basename = ".".join(os.path.basename(filenames).split(".")[:-1])
    output_basename += ".zarr"
    output_filename = os.path.join(output_dir, output_basename)

    # Load the metadata from the input image to define the size of the output
    # zarr arrays.
    image_loader = zds.ImageLoader(
        filename=filenames,
        source_axes=source_axes,
        data_group=data_group,
        roi=roi,
        axes=axes
    )

    image_shape = image_loader.shape

    if output_dtype is None:
        output_dtype = image_loader.arr.dtype

    if output_axes is None:
        output_axes = image_loader.axes

    output_shape = [
        round(scale * image_shape[image_loader.axes.index(ax)])
        for ax in output_axes
        if ax in image_loader.axes
    ]

    output_chunks = [
        patch_size if ax in "ZYX" else 1
        for ax in output_axes
    ]

    out_grp = zarr.open(output_filename, mode="a")

    out_grp.create_dataset(
        data=data,
        name=output_array_name,
        shape=output_shape,
        chunks=output_chunks,
        compressor=zarr.Blosc(clevel=9),
        dtype=output_dtype,
        write_empty_chunks=False,
        overwrite=True
    )

    if output_modality == "labels":
        dataspecs_class = zds.LabelsDatasetSpecs
    elif output_modality == "masks":
        dataspecs_class = zds.MasksDatasetSpecs
    else:
        dataspecs_class = zds.ImagesDatasetSpecs

    output_metadata = dataspecs_class(
        filenames=output_filename,
        source_axes=output_axes,
        axes=output_axes,
        data_group=output_array_name,
        roi=slice(None)
    )

    return output_metadata
