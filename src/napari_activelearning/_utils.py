from typing import Optional, Union, Iterable
from pathlib import PureWindowsPath, Path
from urllib.parse import urlparse
import random
import math
import zarr
import tifffile
import zarrdataset as zds
from ome_zarr.writer import write_multiscales_metadata, write_label_metadata
from ome_zarr.format import FormatV04
import dask.array as da

from skimage.transform import resize
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    USING_PYTORCH = True
except ModuleNotFoundError:
    USING_PYTORCH = False

from napari.layers import Layer
from napari.layers._multiscale_data import MultiScaleData

from ._models import TunableMethod, AxesCorrector


class SuperPixelGenerator(zds.MaskGenerator):
    """Gerates a labeled mask based on the super pixels computed from the input
    image. The super pixels are computed using the SEEDS method.

    The axes of the input image are expected to be YXC, or YX if the image has
    no channels.
    """
    def __init__(self, num_superpixels: int = 512, axes: str = "YXC",
                 model_axes: str = "YXC",
                 **kwargs):

        super(SuperPixelGenerator, self).__init__(axes=axes)

        self._num_superpixels = num_superpixels
        self._model_axes = model_axes
        self._model_spatial_axes = self._model_axes
        if "C" in self._model_spatial_axes:
            self._model_spatial_axes = list(self._model_spatial_axes)
            self._model_spatial_axes.remove("C")
            self._model_spatial_axes = "".join(self._model_spatial_axes)

        self._pos = tuple(
            slice(None) if ax in self._model_spatial_axes else None
            for ax in self.axes
        )

    def _compute_transform(self, image):
        image_shape = {
            ax: ax_s
            for ax, ax_s in zip(self.axes, image.shape)
        }

        spatial_size = tuple(
            ax_s
            for ax, ax_s in image_shape.items()
            if ax in self._model_spatial_axes
        )

        scales = np.array(tuple(
            ax_s / sum(spatial_size)
            for ax, ax_s in image_shape.items()
            if ax in self._model_spatial_axes
        ))

        dim_sizes = self._num_superpixels ** scales
        dim_sizes = dim_sizes.astype(np.int32)

        labels_dim = np.arange(np.prod(dim_sizes)).reshape(dim_sizes)
        labels = resize(labels_dim, spatial_size, order=0)

        labels = labels[self._pos]
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
            for ax, chk in zip(image.source_axes, image.arr.chunks)
            if ax in self.spatial_axes
        }

        image_size = {
            ax: s
            for ax, s in zip(image.source_axes, image.arr.shape)
        }

        self._max_chunk_size = {
            ax: (min(max(self._max_chunk_size[ax],
                         spatial_chunk_sizes[ax]),
                     image_size[ax]))
            if ax in image.source_axes else 1
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
            np.split(valid_mask_toplefts, len(self.spatial_axes), axis=-1),
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

        image_size = {
            ax: s
            for ax, s in zip(image.source_axes, image.arr.shape)
        }

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


def get_dataloader(
        dataset_metadata: dict,
        patch_size: dict,
        shuffle: bool = True,
        num_workers: int = 0,
        batch_size: int = 1,
        spatial_axes: str = "YX",
        padding: dict = None,
        model_input_axes: str = "YXC",
        tunable_segmentation_method: Optional[TunableMethod] = None,
        **superpixel_kwargs
):

    if "superpixels" not in dataset_metadata:
        dataset_metadata["superpixels"] = zds.ImagesDatasetSpecs(
            filenames=dataset_metadata["images"]["filenames"],
            data_group=dataset_metadata["images"]["data_group"],
            source_axes=dataset_metadata["images"]["source_axes"],
            axes=dataset_metadata["images"]["axes"],
            roi=dataset_metadata["images"]["roi"],
            image_loader_func=SuperPixelGenerator(
                axes=dataset_metadata["images"]["axes"],
                model_axes=model_input_axes,
                **superpixel_kwargs
            ),
            modality="superpixels"
        )

    patch_sampler = zds.PatchSampler(patch_size=patch_size,
                                     spatial_axes=spatial_axes,
                                     pad=padding,
                                     min_area=1)

    train_dataset = zds.ZarrDataset(
        list(dataset_metadata.values()),
        return_positions=True,
        draw_same_chunk=False,
        patch_sampler=patch_sampler,
        shuffle=shuffle
    )

    if tunable_segmentation_method is not None:
        base_mode_transforms =\
            tunable_segmentation_method.get_inference_transform()

        if base_mode_transforms is None:
            base_mode_transforms = {}

        base_mode_transforms = {
            input_mode: [mode_transforms]
            for input_mode, mode_transforms in
            base_mode_transforms.items()
        }

        # Complete the transforms for individial input modes
        mode_transforms = {
            (input_mode, ): []
            for input_mode in dataset_metadata.keys()
            if (input_mode, ) not in base_mode_transforms
        }
        mode_transforms.update(base_mode_transforms)

        for input_mode in dataset_metadata.keys():
            mode_transforms[(input_mode, )].insert(0, AxesCorrector(
                dataset_metadata[input_mode]["axes"],
                model_input_axes
            ))

        for input_mode, transform_mode in mode_transforms.items():
            train_dataset.add_transform(input_mode, transform_mode)

    if USING_PYTORCH:
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
              is_label: bool = False,
              overwrite: bool = True):
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

    if overwrite:
        group_name = name
    else:
        group_name = get_next_name(name, list(out_grp.keys()))

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
            dimension_separator="/",  # To make it compatible with ome-zarr
            dtype=dtype if data_ms_s is None else data_ms_s.dtype,
            overwrite=True
        )

    if (is_label and isinstance(out_grp.store, (zarr.MemoryStore,
                                                zarr.DirectoryStore))):
        write_label_metadata(out_grp, group_name, fmt=FormatV04(), **metadata)

    return out_grp, group_name


def update_labels(labels_group: zarr.Group,
                  new_labels: Union[np.ndarray, set]):
    label_colors = labels_group.attrs["image-label"].get("colors", [])

    prev_unique_labels = set(
        lbl["label-value"]
        for lbl in label_colors
    )

    # Remove existing label values from the new labels list and
    # the background label as well.
    if isinstance(new_labels, np.ndarray):
        new_labels = set(np.unique(new_labels))

    unique_labels = new_labels - prev_unique_labels - {0}

    if len(unique_labels):
        label_colors += [
            {
                "label-value": label_value,
                "rgba": [
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                    255
                ],
            }
            for label_value in unique_labels
        ]

        label_version = labels_group.attrs["image-label"]["version"]
        labels_group.attrs["image-label"] = {
            "version": label_version,
            "colors": label_colors
        }


def downsample_image(z_root, axes, scale, data_group,
                     downsample_scale=4,
                     num_scales=5,
                     reference_source_axes=None,
                     reference_scale=None,
                     reference_units=None):
    if reference_source_axes is None or reference_scale is None:
        reference_source_axes = axes
        reference_scale = {
            ax: 1.0
            for ax in axes
        }

    datasets = [{
        "coordinateTransformations": [{
            "type": "scale",
            "scale": [scale.get(ax, 1) * reference_scale.get(ax, 1.0)
                      for ax in axes],
            "translation": [
                0.5 * (scale.get(ax, 1) * reference_scale.get(ax) - 1)
                if (ax in "YX"
                    and (scale.get(ax, 1) * reference_scale.get(ax)) > 1)
                else 0.0
                for ax in axes
            ]
        }],
        "path": "0"
    }]

    z_ms = None

    if num_scales > 1:
        if isinstance(z_root, (Path, str)):
            source_arr = da.from_zarr(z_root, component=data_group)
            z_ms = [source_arr]

        elif isinstance(z_root, np.ndarray):
            source_arr = da.from_array(z_root)
            z_ms = [source_arr]

        else:
            source_arr = da.from_zarr(z_root[data_group])
            z_ms = [source_arr]

        if data_group is None:
            data_group = ""
        else:
            data_group_parts = Path(data_group).parts
            if len(data_group_parts) > 1:
                data_group = str(Path(*data_group_parts[:-1]))

        groups_root = data_group + "/%i"

        source_arr_shape = {ax: source_arr.shape[axes.index(ax)]
                            for ax in axes}

        min_spatial_shape = min(source_arr_shape[ax]
                                for ax in "YX" if ax in axes)

        num_scales = min(num_scales, int(np.log(min_spatial_shape)
                                         / np.log(downsample_scale)))

        downscale_selection = tuple(
            slice(None, None, downsample_scale)
            if ax in "YX" and ax_s > 1
            else slice(None)
            for ax, ax_s in zip(axes, source_arr.shape)
        )

        if not reference_units:
            reference_units = {
                ax: None
                for ax in reference_source_axes
            }

        for s in range(1, num_scales):
            target_arr = source_arr[downscale_selection]
            # Don't generate chunks smaller than 256 per side or the current
            # size of the source array.
            target_arr = target_arr.rechunk(
                tuple(
                    max(min(256, chk),
                        chk // (scale.get(ax, 1) * reference_scale.get(ax, 1)))
                    if ax in "XY" else chk
                    for chk, ax in zip(source_arr.chunksize, axes)
                )
            )

            if isinstance(z_root, (Path, str)):
                z_ms.append(target_arr)

                target_arr.to_zarr(
                    z_root,
                    component=groups_root % s,
                    compressor=zarr.Blosc(clevel=9),
                    write_empty_chunks=False,
                    dimension_separator="/",
                    overwrite=True
                )

                source_arr = da.from_zarr(z_root, component=groups_root % s)
                datasets.append({
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": [
                                (float(downsample_scale) ** s
                                 * scale.get(ax, 1.0)
                                 * reference_scale.get(ax, 1.0))
                                if ax in "YX" else 1.0
                                for ax in axes
                            ],
                            "translation": [
                                0.5 * (float(downsample_scale) ** s
                                       * scale.get(ax, 1.0)
                                       * reference_scale.get(ax, 1.0) - 1)
                                if (ax in "YX"
                                    and (scale.get(ax, 1)
                                         * reference_scale.get(ax, 1)) > 1)
                                else 0
                                for ax in axes
                            ]
                        }
                    ],
                    "path": str(s)
                })

            else:
                z_ms.append(target_arr)
                source_arr = target_arr

    if isinstance(z_root, Path):
        z_grp = zarr.open(z_root / data_group, mode="a")
        write_multiscales_metadata(z_grp, datasets,
                                   fmt=FormatV04(),
                                   name=data_group,
                                   axes=list(axes.lower()))

    return z_ms


def get_source_data(layer: Layer, data_group_init: Optional[str] = None):
    if data_group_init is None:
        data_group_init = ""

    input_filename = layer._source.path
    data_group = data_group_init
    available_data_groups = []

    if input_filename:
        input_url = urlparse(input_filename)

        input_scheme = input_url.scheme
        input_netloc = input_url.netloc
        input_path = Path(input_url.path)

        input_filename_parts = input_path.parts
        extension_idx = list(filter(lambda idx:
                                    ".zarr" in input_filename_parts[idx],
                                    range(len(input_filename_parts))))
        if extension_idx:
            extension_idx = extension_idx[0]
            data_group_filename = input_filename_parts[extension_idx + 1:]
            if len(data_group_filename):
                data_group_filename = Path(*data_group_filename)
            else:
                data_group_filename = ""

            data_group = data_group_filename if not data_group else data_group

            input_path = Path(
                *input_filename_parts[:extension_idx + 1]
            )

        if isinstance(data_group, PureWindowsPath):
            data_group = data_group.as_posix()

        if isinstance(input_path, PureWindowsPath):
            input_path = input_path.as_posix()

        data_group = str(data_group)
        input_path = str(input_path)

        if input_scheme:
            if input_scheme in ["http", "https", "ftp", "s3"]:
                input_scheme += "://"
            else:
                input_scheme += ":"

        input_filename = input_scheme + input_netloc + input_path

        zarr_like = False
        if ".zarr" in input_filename:
            zarr_like = True
            z_fp = input_filename
        else:
            # Try to read as a TIFF file if not a Zarr
            try:
                z_fp = tifffile.imread(input_filename, aszarr=True)
                zarr_like = True
            except tifffile.TiffFileError:
                zarr_like = False

        if zarr_like:
            # Set up the Zarr group
            z_grp = zarr.open(z_fp, mode="r")
            if not isinstance(z_grp, zarr.Array):
                parent_group = ""
                while True:
                    if (data_group not in z_grp
                       or isinstance(z_grp[data_group], zarr.Array)):
                        break
                    parent_group = data_group
                    data_group = str(Path(data_group) / "0")

                available_data_groups = []
                for group_name in range(10):
                    try:
                        z_grp[parent_group][group_name].shape
                        available_data_groups.append(
                            str(Path(parent_group) / str(group_name))
                        )
                    except AttributeError:
                        break
                    except KeyError:
                        break

    else:
        return layer.data, None, []

    if not input_filename:
        input_filename = None

    if not data_group:
        data_group = None

    return input_filename, data_group, available_data_groups
