from typing import Optional, Union, Iterable
from pathlib import Path

import math
import tensorstore as ts
import zarr
import zarrdataset as zds
from ome_zarr.writer import write_multiscales_metadata, write_label_metadata
import dask.array as da

try:
    import cv2
    from cv2.ximgproc import createSuperpixelSEEDS
    CV_SUPERPIXELS = True
except ModuleNotFoundError:
    from skimage.transform import resize
    CV_SUPERPIXELS = False

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    USING_TORCH = True
except ModuleNotFoundError:
    USING_TORCH = False


from napari.layers import Layer
from napari.layers._multiscale_data import MultiScaleData


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
                 double_step: bool = False,
                 axes: str = "YXC"):

        super(SuperPixelGenerator, self).__init__(axes=axes)
        self._iterations = iterations
        self._prior = prior
        self._num_superpixels = num_superpixels
        self._num_levels = num_levels
        self._histogram_bins = histogram_bins
        self._double_step = double_step

        self._ax_X = self.axes.index("X")
        self._ax_Y = self.axes.index("Y")
        if "C" in self.axes:
            self._ax_C = self.axes.index("C")
        else:
            self._ax_C = None

    def _compute_transform(self, image):
        if image.ndim > 2:
            image_channels = image.shape[self._ax_C]
        else:
            image_channels = 1

        if CV_SUPERPIXELS:
            super_pixels = createSuperpixelSEEDS(
                image.shape[self._ax_Y],
                image.shape[self._ax_X],
                image_channels=image_channels,
                prior=self._prior,
                num_superpixels=self._num_superpixels,
                num_levels=self._num_levels,
                histogram_bins=self._histogram_bins,
                double_step=self._double_step
            )

            norm_image = np.stack((image), axis=2)

            norm_image = (
                (np.copy(image) - image.min(axis=(self._ax_Y, self._ax_X),
                                            keepdims=True))
                / (image.max(axis=(self._ax_Y, self._ax_X), keepdims=True)
                   - image.min(axis=(self._ax_Y, self._ax_X), keepdims=True))
            )

            norm_image = 255.0 * norm_image
            norm_image = norm_image.astype(np.uint8)

            super_pixels.iterate(norm_image, self._iterations)
            labels = super_pixels.getLabels()

        else:
            cols = int(np.sqrt(self._num_superpixels))
            rows = self._num_superpixels // cols

            labels_dim = np.arange(cols * rows).reshape(rows, cols)
            labels = resize(labels_dim,
                            (image.shape[self.ax_Y], image.shape[self.ax_X]),
                            order=0)

        if image_channels > 1:
            labels = np.expand_dims(labels, self._ax_C)
        else:
            labels = labels[..., None]

        return labels


class StaticPatchSampler(zds.PatchSampler):
    """Static patch sampler that retrieves patches pre-defined positions.

    Parameters
    ----------
    patch_size : int, iterable, dict
        Size in pixels of the patches extracted. Only squared patches are
        supported by now.
    top_lefts : Iterable[Iterable[int]]
        A list of top-left postions to sample.
    """
    def __init__(self, patch_size: Union[int, Iterable[int], dict],
                 top_lefts: Iterable[Iterable[int]],
                 **kwargs):
        super(StaticPatchSampler, self).__init__(patch_size, **kwargs)
        self._top_lefts = np.array(top_lefts)

    def compute_chunks(self, image_collection: zds.ImageCollection
                       ) -> Iterable[dict]:
        image = image_collection.collection["images"]

        spatial_chunk_sizes = {
            ax: (self._stride[ax]
                 * max(1, math.ceil(chk / self._stride[ax])))
            for ax, chk in zip(image.axes, image.chunk_size)
            if ax in self.spatial_axes
        }

        image_size = {ax: s for ax, s in zip(image.axes, image.shape)}

        self._max_chunk_size = {
            ax: (min(max(self._max_chunk_size[ax],
                         spatial_chunk_sizes[ax]),
                     image_size[ax]))
            if ax in image.axes else 1
            for ax in self.spatial_axes
        }

        valid_mask_toplefts = np.array([
            [ax_tl // spatial_chunk_sizes.get(ax, 1)
             for ax_tl, ax in zip(tl, self.spatial_axes)]
            for tl in self._top_lefts
        ])

        num_blocks = [
            int(math.ceil(image_size.get(ax, 1)
                          / spatial_chunk_sizes.get(ax, 1)))
            for ax in self.spatial_axes
        ]

        valid_mask_toplefts = np.ravel_multi_index(
            np.split(valid_mask_toplefts, 2, axis=1),
            num_blocks
        )
        valid_mask_toplefts = np.unique(valid_mask_toplefts)
        valid_mask_toplefts = np.unravel_index(valid_mask_toplefts, num_blocks)
        valid_mask_toplefts = tuple(
            indices.reshape(-1, 1)
            for indices in valid_mask_toplefts
        )
        valid_mask_toplefts = np.hstack(valid_mask_toplefts)

        spatial_chunk_sizes_arr = np.array([[
            spatial_chunk_sizes.get(ax, 1)
            for ax in self.spatial_axes
        ]])

        valid_mask_toplefts = valid_mask_toplefts * spatial_chunk_sizes_arr

        chunk_tlbr = {ax: slice(None) for ax in self.spatial_axes}

        chunks_slices = self._compute_toplefts_slices(
            chunk_tlbr,
            valid_mask_toplefts=valid_mask_toplefts,
            patch_size=self._max_chunk_size
        )

        return chunks_slices

    def compute_patches(self, image_collection: zds.ImageCollection,
                        chunk_tlbr: dict) -> Iterable[dict]:
        image = image_collection.collection[image_collection.reference_mode]

        image_size = {ax: s for ax, s in zip(image.axes, image.shape)}

        patch_size = {
            ax: self._patch_size.get(ax, 1) if image_size.get(ax, 1) > 1 else 1
            for ax in self.spatial_axes
        }

        pad = {
            ax: self._pad.get(ax, 0) if image_size.get(ax, 1) > 1 else 0
            for ax in self.spatial_axes
        }

        min_area = self._min_area
        if min_area < 1:
            min_area *= np.prod(list(patch_size.values()))

        chunk_tl_limit = np.array(list(
            map(lambda chk_slice:
                chk_slice.start if chk_slice.start is not None else 0,
                [chunk_tlbr.get(ax, slice(None)) for ax in self.spatial_axes])
        ))

        chunk_br_limit = np.array(list(
            map(lambda chk_slice:
                chk_slice.stop if chk_slice.stop is not None else float("inf"),
                [chunk_tlbr.get(ax, slice(None)) for ax in self.spatial_axes])
        ))

        valid_mask_toplefts_idx = np.bitwise_and(
            self._top_lefts >= chunk_tl_limit[None, ...],
            self._top_lefts < chunk_br_limit[None, ...]
        )
        valid_mask_toplefts_idx = np.all(valid_mask_toplefts_idx, axis=1)

        valid_mask_toplefts = self._top_lefts[valid_mask_toplefts_idx]
        valid_mask_toplefts = valid_mask_toplefts - chunk_tl_limit[None, ...]

        patches_slices = self._compute_toplefts_slices(
            chunk_tlbr,
            valid_mask_toplefts=valid_mask_toplefts,
            patch_size=patch_size,
            pad=pad
        )

        return patches_slices


def get_dataloader(dataset_metadata, patch_size=512,
                   sampling_positions=None,
                   shuffle=True,
                   num_workers=0,
                   batch_size=1,
                   spatial_axes="YX",
                   **superpixel_kwargs):

    if "superpixels" not in dataset_metadata:
        dataset_metadata["superpixels"] = zds.ImagesDatasetSpecs(
            filenames=dataset_metadata["images"]["filenames"],
            data_group=dataset_metadata["images"]["data_group"],
            source_axes=dataset_metadata["images"]["source_axes"],
            axes=dataset_metadata["images"]["axes"],
            roi=dataset_metadata["images"]["roi"],
            image_loader_func=SuperPixelGenerator(
                axes=dataset_metadata["images"]["axes"],
                **superpixel_kwargs
            ),
            modality="superpixels"
        )

    if sampling_positions:
        patch_sampler = StaticPatchSampler(patch_size=patch_size,
                                           top_lefts=sampling_positions,
                                           spatial_axes=spatial_axes)
    else:
        patch_sampler = zds.PatchSampler(patch_size=patch_size,
                                         spatial_axes=spatial_axes,
                                         min_area=0.25)

    train_dataset = zds.ZarrDataset(
        list(dataset_metadata.values()),
        return_positions=True,
        draw_same_chunk=True,
        patch_sampler=patch_sampler,
        shuffle=shuffle
    )

    if USING_TORCH:
        train_dataloader = DataLoader(
            train_dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=zds.zarrdataset_worker_init_fn
        )
    else:
        train_dataloader = train_dataset

    return train_dataloader


def get_basename(layer_name):
    layer_base_name = layer_name.split(" ")
    if len(layer_base_name) > 1:
        layer_base_name = " ".join(layer_base_name[:-1])
    else:
        layer_base_name = layer_base_name[0]

    return layer_base_name


def get_next_name(name: str, group_names: Iterable[str]):
    if name in group_names:
        n_existing = sum(map(
            lambda exisitng_group_name:
            name in exisitng_group_name,
            group_names
        ))

        new_name = name + " (%i)" % n_existing
    else:
        new_name = name

    return new_name


def validate_name(group_names, previous_child_name, new_child_name):
    if previous_child_name in group_names:
        group_names.remove(previous_child_name)

    if new_child_name:
        new_child_name = get_next_name(new_child_name, group_names)
        group_names.add(new_child_name)

    return new_child_name


def save_zarr(output_filename, data, shape, chunk_size, name, dtype,
              is_multiscale: bool = False,
              metadata: Optional[dict] = None,
              is_label: bool = False):
    if not metadata:
        metadata = {}

    if output_filename is None:
        out_grp = zarr.group()
    elif isinstance(output_filename, (Path, str)):
        out_grp = zarr.open(output_filename, mode="a")
    elif isinstance(output_filename, zarr.Group):
        out_grp = output_filename
    else:
        raise ValueError(f"Output filename of type {type(output_filename)} is"
                         f" not supported")

    if not isinstance(chunk_size, bool) and isinstance(chunk_size, int):
        chunk_size = [chunk_size] * len(shape)

    if isinstance(chunk_size, list):
        chunks_size_axes = list(map(min, shape, chunk_size))
    else:
        chunks_size_axes = chunk_size

    group_name = name
    if is_label:
        group_name = "labels/" + group_name

    if isinstance(data, MultiScaleData):
        data_ms = data
    else:
        data_ms = [data]

    num_scales = len(data_ms)

    if num_scales > 1:
        group_ms_names = [
            group_name + ("/%i" % s if is_multiscale else "")
            for s in range(num_scales)
        ]
    else:
        group_ms_names = [group_name + ("/0" if is_multiscale else "")]

    for data_ms_s, group_ms_s in zip(data_ms, group_ms_names):

        if data_ms_s is not None and not isinstance(data_ms_s, np.ndarray):
            data_ms_s = np.array(data_ms_s)

        out_grp.create_dataset(
            data=data_ms_s,
            name=group_ms_s,
            shape=shape if data_ms_s is None else data_ms_s.shape,
            chunks=chunks_size_axes,
            compressor=zarr.Blosc(clevel=9),
            write_empty_chunks=False,
            dtype=dtype if data_ms_s is None else data_ms_s.dtype,
            overwrite=True
        )

    if not isinstance(out_grp.store, zarr.MemoryStore):
        write_label_metadata(out_grp, group_name, **metadata)

    return out_grp


def downsample_image(z_root, source_axes, data_group, scale=4, num_scales=5,
                     reference_source_axes=None,
                     reference_scale=None,
                     reference_units=None):
    if isinstance(z_root, (Path, str)):
        source_arr = da.from_zarr(z_root, component=data_group)

        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': str(Path(z_root) / data_group),
            },
        }

        ts_array = ts.open(spec).result()
        z_ms = [ts_array]

    elif isinstance(z_root, np.ndarray):
        source_arr = da.from_array(z_root)

    else:
        source_arr = da.from_zarr(z_root[data_group])
        z_ms = [source_arr]

    data_group = "/".join(data_group.split("/")[:-1])
    groups_root = data_group + "/%i"

    source_arr_shape = {ax: source_arr.shape[source_axes.index(ax)]
                        for ax in source_axes}

    min_spatial_shape = min(source_arr_shape[ax]
                            for ax in "YX" if ax in source_axes)

    num_scales = min(num_scales, int(np.log(min_spatial_shape)
                                     / np.log(scale)))

    downscale_selection = tuple(
        slice(None, None, scale) if ax in "ZYX" and ax_s > 1 else slice(None)
        for ax, ax_s in zip(source_axes, source_arr.shape)
    )

    if reference_source_axes is None or reference_scale is None:
        reference_source_axes = source_axes
        reference_scale = [1.0] * len(source_axes)

    reference_scale_axes = {
        ax: ax_scl
        for ax, ax_scl in zip(reference_source_axes, reference_scale)
    }

    if not reference_units:
        reference_units = {
            ax: None
            for ax in reference_source_axes
        }

    datasets = [{
        "coordinateTransformations": [{
            "type": "scale",
            "scale": [reference_scale_axes.get(ax, 1.0) for ax in source_axes]
        }],
        "path": "0"
    }]

    for s in range(1, num_scales):
        target_arr = source_arr[downscale_selection]
        target_arr = target_arr.rechunk()

        if isinstance(z_root, (Path, str)):
            z_ms.append(z_ms[-1][downscale_selection])

            target_arr.to_zarr(z_root,
                               component=groups_root % s,
                               compressor=zarr.Blosc(clevel=9),
                               write_empty_chunks=False,
                               overwrite=True)

            source_arr = da.from_zarr(z_root, component=groups_root % s)

            datasets.append({
                "coordinateTransformations": [
                    {"type": "scale",
                     "scale": [4.0 ** s * reference_scale_axes.get(ax, 1.0)
                               if ax in "ZYX" else 1.0
                               for ax in source_axes]
                     }],
                "path": str(s)
            })

        else:
            z_ms.append(target_arr)
            source_arr = target_arr

    if isinstance(z_root, Path):
        z_grp = zarr.open(z_root / data_group, mode="a")
        write_multiscales_metadata(z_grp, datasets,
                                   axes=list(source_axes.lower()))

    return z_ms


def get_source_data(layer: Layer):
    input_filename = layer._source.path
    data_group = ""

    if input_filename:
        input_filename = str(input_filename)
        data_group = "/".join(input_filename.split(".")[-1].split("/")[1:])
    else:
        return layer.data, None

    if data_group:
        input_filename = input_filename[:-len(data_group) - 1]

    if input_filename and isinstance(layer.data, MultiScaleData):
        data_group = str(Path(data_group) / "0")

    if not input_filename:
        input_filename = None

    if not data_group:
        data_group = None

    return input_filename, data_group
