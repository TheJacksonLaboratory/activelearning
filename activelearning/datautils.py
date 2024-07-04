from typing import Optional, Union, Iterable
import os
import math
import napari
import tensorstore as ts
import zarr
import zarrdataset as zds
import dask.array as da
from dask.diagnostics import ProgressBar
import cv2
import numpy as np
from functools import partial, reduce

import torch
from torch.utils.data import DataLoader


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
            image_channels = image.shape[-1]
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
            image_size.get(ax, 1) // spatial_chunk_sizes.get(ax, 1)
            for ax in self.spatial_axes
        ]

        valid_mask_toplefts = np.ravel_multi_index(
            np.split(valid_mask_toplefts, 2),
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
        valid_mask_toplefts_idx = np.all(valid_mask_toplefts_idx, axis=0)

        valid_mask_toplefts = self._top_lefts[valid_mask_toplefts_idx]

        patches_slices = self._compute_toplefts_slices(
            chunk_tlbr,
            valid_mask_toplefts=valid_mask_toplefts,
            patch_size=patch_size,
            pad=pad
        )

        return patches_slices



def get_dataloader(dataset_metadata, patch_size=512,
                   sampling_positions_dict=None,
                   shuffle=True,
                   num_workers=0,
                   batch_size=1,
                   **superpixel_kwargs):

    if "superpixels" not in dataset_metadata:
        dataset_metadata["superpixels"] = zds.ImagesDatasetSpecs(
            filenames=dataset_metadata["images"]["filenames"],
            data_group=dataset_metadata["images"]["data_group"],
            source_axes=dataset_metadata["images"]["source_axes"],
            axes=dataset_metadata["images"]["axes"],
            roi=dataset_metadata["images"]["roi"],
            image_loader_func=SuperPixelGenerator(**superpixel_kwargs),
            modality="superpixels"
        )

    if "masks" in dataset_metadata:
        spatial_axes = dataset_metadata["masks"]["axes"]

    else:
        spatial_axes = "YX"

    if sampling_positions_dict:
        patch_sampler = StaticPatchSampler(patch_size=patch_size,
                                           top_lefts=sampling_positions_dict,
                                           spatial_axes=spatial_axes)
    else:
        patch_sampler = zds.PatchSampler(patch_size=patch_size,
                                         spatial_axes=spatial_axes,
                                         min_area=0.25)

    train_dataset = zds.ZarrDataset(
        list(dataset_metadata.values()),
        return_positions=True,
        draw_same_chunk=False,
        patch_sampler=patch_sampler,
        shuffle=shuffle
    )

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=zds.zarrdataset_worker_init_fn
    )

    return train_dataloader
