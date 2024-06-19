import os
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


def get_dataloader(dataset_metadata, patch_size=512, shuffle=True,
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

    train_dataset = zds.ZarrDataset(
        list(dataset_metadata.values()),
        return_positions=True,
        draw_same_chunk=False,
        patch_sampler=zds.PatchSampler(patch_size=patch_size,
                                       spatial_axes=spatial_axes,
                                       min_area=0.25),
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
