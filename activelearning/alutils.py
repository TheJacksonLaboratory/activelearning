import os
from typing import List, Tuple, Union
import zarrdataset as zds
import torch
from torch.utils.data import DataLoader
import numpy as np
import zarr
import tqdm

import datautils



def compute_acquisition_image(model, inference_fn, output_dir, image_metadata,
                              image_index=None,
                              patch_size=256,
                              samples_per_image=None,
                              repetitions=30,
                              num_workers=0,
                              **inference_fn_args):
    train_dataset = datautils.get_zarrdataset(image_metadata,
                                              patch_size=patch_size)


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

        if samples_per_image is not None and n_samples >= samples_per_image:
            break

    return samples_u_img
