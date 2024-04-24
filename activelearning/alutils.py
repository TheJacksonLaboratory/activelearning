from typing import List, Tuple, Union
import zarrdataset as zds
import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import dask.array as da
import zarr
import napari

from itertools import cycle

import datautils


class DropoutEvalOverrider(torch.nn.Module):
    def __init__(self, dropout_module):
        super(DropoutEvalOverrider, self).__init__()

        self._dropout = type(dropout_module)(dropout_module.p,
                                             inplace=dropout_module.inplace)

    def forward(self, input):
        training_temp = self._dropout.training

        self._dropout.training = True
        out = self._dropout(input)

        self._dropout.training = training_temp

        return out


def add_dropout(net, p=0.05):
    # First step checks if there is any Dropout layer existing in the model
    has_dropout = False
    for module in net.modules():
        if isinstance(module, torch.nn.Sequential):
            for l_idx, layer in enumerate(module):
                if isinstance(layer, (torch.nn.Dropout, torch.nn.Dropout1d,
                                      torch.nn.Dropout2d,
                                      torch.nn.Dropout3d)):
                    has_dropout = True
                    break
            else:
                continue

            dropout_layer = module.pop(l_idx)
            module.insert(l_idx, DropoutEvalOverrider(dropout_layer))

    if has_dropout:
        return

    for module in net.modules():
        if isinstance(module, torch.nn.Sequential):
            for l_idx, layer in enumerate(module):
                if isinstance(layer, torch.nn.ReLU):
                    break
            else:
                continue

            dropout_layer = torch.nn.Dropout(p=p, inplace=True)
            module.insert(l_idx + 1, DropoutEvalOverrider(dropout_layer))


def annotate_samples(active_samples, patch_size, num_workers):
    samples_imgs = []
    samples_u_sps = []

    for image_metadata, output_filename, _ in active_samples:
        labels = datautils.annotate_mask(
            dict(input_image=(image_metadata, True)),
                 patch_size=args.patch_size
            )
        )

        annotate_mask(layers_metadata, patch_size=256, **kwargs)

    samples_imgs_da = da.stack(samples_imgs, axis=0)
    samples_u_sps_da = da.stack(samples_u_sps, axis=0)
    samples_annotations = zarr.zeros(samples_u_sps_da.shape, dtype=np.int64)

    viewer = napari.Viewer(title="Phase III: Annotating samples")
    viewer.add_image(samples_imgs_da, rgb=True, name="Inputs")
    viewer.add_image(samples_u_sps_da, name="Uncertainity heatmap", opacity=0.35, colormap="inferno")
    viewer.add_labels(samples_annotations, name="New annotations")

    napari.run()

    curr_idx = 0
    for image_metadata, output_filename, rois in active_samples.values():
        curr_annotations = zarr.open(output_filename, mode="a")["annotations/0"]
        for roi in rois:
            ann_roi = tuple(roi[image_metadata["source_axes"].index(ax)] for ax in "YX")
            curr_annotations[ann_roi] = samples_annotations[curr_idx, ...]
            curr_idx += 1


def compute_BALD(probs):
    if probs.ndim == 3:
        probs = np.stack((probs, 1 - probs), axis=1)

    T = probs.shape[0]

    probs_mean = probs.mean(axis=0)

    mutual_info = (-np.sum(probs_mean * np.log(probs_mean + 1e-12), axis=0)
                   + np.sum(probs * np.log(probs + 1e-12), axis=(0, 1)) / T)

    return mutual_info


def compute_acquisition(probs, super_pixel_labels):
    mutual_info = compute_BALD(probs)

    super_pixel_indices = np.unique(super_pixel_labels)

    u_sp = []
    u_sp_lab = np.zeros_like(super_pixel_labels)

    for sp_l in super_pixel_indices:
        mask = super_pixel_labels == sp_l
        u_val = np.sum(mutual_info[mask]) / np.sum(mask)

        u_sp.append(u_val)
        u_sp_lab = np.where(mask, u_val, u_sp_lab)

    return u_sp, u_sp_lab


def compute_confidence_image(model, inference_fn, image_metadata, mask_metadata, output_filename, image_index=None, patch_size=256, samples_per_image=None, repetitions=30, num_workers=0, **inference_fn_args):
    train_dataset = datautils.get_zarrdataset(image_metadata, mask_metadata=mask_metadata, patch_size=patch_size)
    train_dataloader = DataLoader(train_dataset, num_workers=num_workers, batch_size=1, pin_memory=torch.cuda.is_available(), worker_init_fn=zds.zarrdataset_worker_init_fn)

    confidence_map = zarr.open(output_filename, mode="a")["confidence_map/0"]

    n_samples = 0
    samples_u_img = []

    for i, (pos, img, img_sp) in tqdm.tqdm(enumerate(train_dataloader), position=1, leave=False, total=samples_per_image, desc="Acquiring samples"):
        probs = []
        for _ in tqdm.tqdm(range(repetitions), desc=f"Computing confidence for sample {i + 1}", position=2, leave=False):
            probs.append(inference_fn(img[0].numpy(), model, **inference_fn_args))

        probs = np.stack(probs, axis=0)

        u_sp, u_sp_lab = compute_acquisition(probs, img_sp[0, ..., 0].numpy())

        pos_u_lab = (slice(pos[0, 0, 0].item(), pos[0, 0, 1].item()), slice(pos[0, 1, 0].item(), pos[0, 1, 1].item()))
        confidence_map[pos_u_lab] = u_sp_lab

        samples_u_img += [
            (u, image_index, pos_u_lab)
            for u in u_sp
        ]

        n_samples += 1
        if samples_per_image is not None and n_samples >= samples_per_image:
            break

    return samples_u_img


def compute_confidence_dataset(model, inference_fn, images_metadata, masks_metadata, output_filenames, patch_size=256, samples_per_image=None, max_samples_to_annotate=None, repetitions=30, num_workers=0, **inference_fn_args):
    samples_u_list = []
    add_dropout(model)

    q = tqdm.tqdm(leave=True, position=0, total=len(images_metadata))
    for image_index, (im_metadata, mk_metadata, output_fiename) in enumerate(zip(images_metadata, masks_metadata, output_filenames)):
        samples_u_list += compute_confidence_image(model, inference_fn, im_metadata, mk_metadata, output_fiename, image_index, patch_size, samples_per_image=samples_per_image, repetitions=repetitions, num_workers=num_workers, **inference_fn_args)
        q.update()

    q.close()
    samples_u_list.sort(reverse=True)
    if max_samples_to_annotate is not None:
        samples_u_list = samples_u_list[:max_samples_to_annotate]

    # Remove the samples that have high confidence from the selected images
    active_samples = []
    active_image_indices = []

    for _, image_index, pos_u_lab in samples_u_list:
        if image_index in active_image_indices:
            curr_sample = active_image_indices.index(image_index)
        else:
            active_image_indices.append(image_index)
            active_samples.append((images_metadata[image_index], output_filenames[image_index], []))
            curr_sample = -1

        curr_image_metadata = images_metadata[image_index]

        base_roi = curr_image_metadata["roi"]
        if not isinstance(base_roi, (tuple, list)):
            base_roi = [base_roi]

        if len(base_roi) < len(curr_image_metadata["source_axes"]):
            base_roi = cycle(base_roi)

        offset_roi = tuple(
            slice((slice_ax.start if slice_ax.start is not None else 0) + pos_u_lab["YX".index(ax)].start,
                  (slice_ax.start if slice_ax.start is not None else 0) + pos_u_lab["YX".index(ax)].stop)
            if ax in "YX" else
            slice_ax
            for ax, slice_ax in zip(curr_image_metadata["source_axes"], base_roi)
        )

        if offset_roi not in active_samples[curr_sample][2]:
            active_samples[curr_sample][2].append(offset_roi)

    return active_samples
