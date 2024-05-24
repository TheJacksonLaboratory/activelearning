import os
import napari
import tensorstore as ts
import zarr
import zarrdataset as zds
import dask.array as da
from dask.diagnostics import ProgressBar
import cv2
import numpy as np
from itertools import repeat
from functools import partial, reduce


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
                                       spatial_axes="YX",
                                       min_area=0.25),
        shuffle=shuffle
    )

    return train_dataset


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

    dataset_metadata = map(partial(zds.parse_metadata,
                                   default_source_axes="TCZYX"),
                           filenames)
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

    if output_dtype is None:
        output_dtype = image_loader.arr.dtype

    if output_axes is None:
        output_axes = image_loader.axes

    output_shape = [
        max(1, round(image_loader.shape[image_loader.axes.index(ax)]
                     * scale))
        for ax in output_axes
        if ax in image_loader.axes
    ]

    output_chunks = [
        max(1, round(image_loader.chunk_size[image_loader.axes.index(ax)]
                     * scale))
        if ax in "ZYX" else 1
        for ax in output_axes
    ]

    return output_metadata


def downsample_chunk(chunk):
    pad = [(0, s % 2) for s in chunk.shape]
    padded_chunk = np.pad(chunk, pad_width=pad)
    downsampled = cv2.resize(padded_chunk, dsize=None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)
    return downsampled


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
