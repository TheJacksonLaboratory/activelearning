import os
from typing import List, Tuple, Union
import zarrdataset as zds
import torch
from torch.utils.data import DataLoader
import numpy as np
import zarr
import tqdm

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
    u_sp_lab = np.zeros_like(super_pixel_labels, dtype=np.float32)

    for sp_l in super_pixel_indices:
        mask = super_pixel_labels == sp_l
        u_val = np.sum(mutual_info[mask]) / np.sum(mask)

        u_sp.append(u_val)
        u_sp_lab = np.where(mask, u_val, u_sp_lab)

    return u_sp, u_sp_lab


def compute_confidence_image(model, inference_fn, output_dir, image_metadata,
                             image_index=None,
                             patch_size=256,
                             samples_per_image=None,
                             repetitions=30,
                             num_workers=0,
                             **inference_fn_args):
    train_dataset = datautils.get_zarrdataset(image_metadata,
                                              patch_size=patch_size)
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        batch_size=1,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=zds.zarrdataset_worker_init_fn
    )

    image_metadata["confidence_maps"] = datautils.prepare_output_zarr(
        "confidence_map/0",
        output_dir=output_dir,
        scale=1,
        output_axes="YX",
        output_dtype=np.float32,
        output_mode="images",
        **image_metadata["images"]
    )

    conf_map = zarr.open(os.path.join(
        image_metadata["confidence_maps"]["filenames"],
        image_metadata["confidence_maps"]["data_group"]),
        mode="a")

    n_samples = 0
    samples_u_img = []

    q_img = tqdm.tqdm(position=1, leave=False, total=samples_per_image,
                      desc="Acquiring samples")
    for i, (pos, img, img_sp) in enumerate(train_dataloader):
        probs = []
        for _ in tqdm.tqdm(range(repetitions),
                           desc=f"Computing confidence for sample {i + 1}",
                           position=2,
                           leave=False):
            probs.append(inference_fn(
                img[0].numpy(), model, **inference_fn_args)
            )

        probs = np.stack(probs, axis=0)

        u_sp, u_sp_lab = compute_acquisition(probs, img_sp[0, ..., 0].numpy())

        pos_u_lab = (slice(pos[0, 0, 0].item(), pos[0, 0, 1].item()),
                     slice(pos[0, 1, 0].item(), pos[0, 1, 1].item()))

        conf_map[pos_u_lab] = u_sp_lab

        samples_u_img += [(u, image_index) for u in u_sp]

        n_samples += 1
        q_img.update()

        if samples_per_image is not None and n_samples >= samples_per_image:
            break

    q_img.close()

    return samples_u_img


def compute_confidence_dataset(model, inference_fn, output_dir,
                               dataset_metadata,
                               patch_size=256,
                               samples_per_image=None,
                               max_samples_to_annotate=None,
                               repetitions=30,
                               num_workers=0,
                               **inference_fn_args):
    samples_u_list = []

    add_dropout(model)

    q = tqdm.tqdm(leave=True, position=0, total=len(dataset_metadata))
    for image_index, image_metadata in enumerate(dataset_metadata):
        samples_u_img = compute_confidence_image(
            model,
            inference_fn,
            output_dir,
            image_metadata,
            image_index,
            patch_size,
            samples_per_image=samples_per_image,
            repetitions=repetitions,
            num_workers=num_workers,
            **inference_fn_args
        )
        samples_u_list += samples_u_img

        q.update()

    q.close()

    samples_u_list.sort(reverse=True)
    if max_samples_to_annotate is not None:
        samples_u_list = samples_u_list[:max_samples_to_annotate]

    samples_u_list = np.array(samples_u_list)[:, 1].astype(np.int32)
    active_samples = np.unique(samples_u_list).tolist()

    active_images = [dataset_metadata[image_index]
                     for image_index in active_samples]

    return active_images
