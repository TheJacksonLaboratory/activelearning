from typing import Iterable, Tuple
import time
import os
import numpy as np
from cellpose import io, transforms, utils, models, dynamics, metrics, resnet_torch, train
from cellpose.transforms import normalize_img
from pathlib import Path
import torch
from torch import nn
from tqdm import trange
from numba import prange

from torch.utils.data import DataLoader, ChainDataset
import random
import zarrdataset as zds
import logging

from datautils import StaticPatchSampler

def train_cellpose(model,
                   dataset_metadata_list: Iterable[Tuple[dict, Iterable[Iterable[int]]]],
                   train_data_proportion: float = 0.8,
                   patch_size: int = 256,
                   spatial_axes="ZYX",
                   **kwargs):

    datasets = [
        zds.ZarrDataset(
            list(dataset_metadata.values()),
            return_positions=False,
            draw_same_chunk=True,
            patch_sampler=StaticPatchSampler(
                patch_size=patch_size,
                top_lefts=top_lefts,
                spatial_axes=spatial_axes
            ),
            shuffle=True,
        )
        for dataset_metadata, top_lefts in dataset_metadata_list
    ]

    chained_datasets = ChainDataset(datasets)

    dataloader = DataLoader(
        chained_datasets,
        num_workers=kwargs.get("num_workers", 0),
        worker_init_fn=zds.chained_zarrdataset_worker_init_fn
    )

    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    for x, t in dataloader:
        lab = t[0].numpy()
        img = x[0].numpy()
        img_t = transforms.convert_image(img, channel_axis=2, channels=[0, 0])
        img_t = transforms.normalize_img(img_t, invert=False, axis=2)

        img_t = img_t.transpose(2, 0, 1)

        if random.random() <= train_data_proportion:
            train_data.append(img_t)
            train_labels.append(lab)
        else:
            test_data.append(img_t)
            test_labels.append(lab)

    if not test_data:
        # Take at least one sample at random from the train dataset
        test_data_idx = random.randrange(0, len(train_data))
        test_data = [train_data.pop(test_data_idx)]
        test_labels = [train_labels.pop(test_data_idx)]

    model_path = train.train_seg(
        model.net,
        train_data=train_data,
        train_labels=train_labels,
        train_probs=None,
        test_data=test_data,
        test_labels=test_labels,
        test_probs=None,
        load_files=True,
        **kwargs
    )

    model = models.CellposeModel(pretrained_model=model_path)

    return model


def cellpose_train_seg(
        net, train_data=None, train_labels=None, train_files=None,
        train_labels_files=None, train_probs=None, test_data=None,
        test_labels=None, test_files=None, test_labels_files=None,
        test_probs=None, load_files=True, batch_size=8, learning_rate=0.005,
        n_epochs=2000, weight_decay=1e-5, momentum=0.9, SGD=False, channels=None,
        channel_axis=None, rgb=False, normalize=True, compute_flows=False,
        save_path=None, save_every=100, nimg_per_epoch=None,
        nimg_test_per_epoch=None, rescale=True, scale_range=None, bsize=224,
        min_train_masks=5, model_name=None):
    """
    Train the network with images for segmentation.

    Args:
        net (object): The network model to train.
        train_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for training. Defaults to None.
        train_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for train_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        train_files (List[str], optional): List of strings - file names for images in train_data (to save flows for future runs). Defaults to None.
        train_labels_files (list or None): List of training label file paths. Defaults to None.
        train_probs (List[float], optional): List of floats - probabilities for each image to be selected during training. Defaults to None.
        test_data (List[np.ndarray], optional): List of arrays (2D or 3D) - images for testing. Defaults to None.
        test_labels (List[np.ndarray], optional): List of arrays (2D or 3D) - labels for test_data, where 0=no masks; 1,2,...=mask labels. Defaults to None.
        test_files (List[str], optional): List of strings - file names for images in test_data (to save flows for future runs). Defaults to None.
        test_labels_files (list or None): List of test label file paths. Defaults to None.
        test_probs (List[float], optional): List of floats - probabilities for each image to be selected during testing. Defaults to None.
        load_files (bool, optional): Boolean - whether to load images and labels from files. Defaults to True.
        batch_size (int, optional): Integer - number of patches to run simultaneously on the GPU. Defaults to 8.
        learning_rate (float or List[float], optional): Float or list/np.ndarray - learning rate for training. Defaults to 0.005.
        n_epochs (int, optional): Integer - number of times to go through the whole training set during training. Defaults to 2000.
        weight_decay (float, optional): Float - weight decay for the optimizer. Defaults to 1e-5.
        momentum (float, optional): Float - momentum for the optimizer. Defaults to 0.9.
        SGD (bool, optional): Boolean - whether to use SGD as optimization instead of RAdam. Defaults to False.
        channels (List[int], optional): List of ints - channels to use for training. Defaults to None.
        channel_axis (int, optional): Integer - axis of the channel dimension in the input data. Defaults to None.
        normalize (bool or dict, optional): Boolean or dictionary - whether to normalize the data. Defaults to True.
        compute_flows (bool, optional): Boolean - whether to compute flows during training. Defaults to False.
        save_path (str, optional): String - where to save the trained model. Defaults to None.
        save_every (int, optional): Integer - save the network every [save_every] epochs. Defaults to 100.
        nimg_per_epoch (int, optional): Integer - minimum number of images to train on per epoch. Defaults to None.
        nimg_test_per_epoch (int, optional): Integer - minimum number of images to test on per epoch. Defaults to None.
        rescale (bool, optional): Boolean - whether or not to rescale images during training. Defaults to True.
        min_train_masks (int, optional): Integer - minimum number of masks an image must have to use in the training set. Defaults to 5.
        model_name (str, optional): String - name of the network. Defaults to None.

    Returns:
        Path: path to saved model weights
    """

    train_logger = logging.getLogger("train")
    train_logger.addHandler(logging.FileHandler("models/training.log", mode='a'))
    train_logger.addHandler(logging.StreamHandler())

    device = net.device

    scale_range0 = 0.5 if rescale else 1.0
    scale_range = scale_range if scale_range is not None else scale_range0

    if isinstance(normalize, dict):
        normalize_params = {**models.normalize_default, **normalize}
    elif not isinstance(normalize, bool):
        raise ValueError("normalize parameter must be a bool or a dict")
    else:
        normalize_params = models.normalize_default
        normalize_params["normalize"] = normalize

    out = train._process_train_test(train_data=train_data, train_labels=train_labels,
                              train_files=train_files, train_probs=train_probs,
                              test_data=test_data, test_labels=test_labels,
                              test_files=test_files, test_probs=test_probs,
                              load_files=load_files, min_train_masks=min_train_masks,
                              compute_flows=compute_flows, channels=channels,
                              channel_axis=channel_axis, rgb=rgb,
                              normalize_params=normalize_params, device=net.device)
    (train_data, train_labels, train_files, train_labels_files, train_probs, diam_train,
     test_data, test_labels, test_files, test_labels_files, test_probs, diam_test,
     normed) = out
    
    dataloader = DataLoader()

    # already normalized, do not normalize during training
    if normed:
        kwargs = {}
    else:
        kwargs = {
            "normalize_params": normalize_params,
            "channels": channels,
            "channel_axis": channel_axis,
            "rgb": rgb
        }

    net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)

    nimg = len(train_data) if train_data is not None else len(train_files)
    nimg_test = len(test_data) if test_data is not None else None
    nimg_test = len(test_files) if test_files is not None else nimg_test
    nimg_per_epoch = nimg if nimg_per_epoch is None else nimg_per_epoch
    nimg_test_per_epoch = nimg_test if nimg_test_per_epoch is None else nimg_test_per_epoch

    # learning rate schedule
    LR = np.linspace(0, learning_rate, 10)
    LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
    if n_epochs > 300:
        LR = LR[:-100]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(10))
    elif n_epochs > 100:
        LR = LR[:-50]
        for i in range(10):
            LR = np.append(LR, LR[-1] / 2 * np.ones(5))

    train_logger.info(f">>> n_epochs={n_epochs}, n_train={nimg}, n_test={nimg_test}")

    if not SGD:
        train_logger.info(
            f">>> AdamW, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}"
        )
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate,
                                      weight_decay=weight_decay)
    else:
        train_logger.info(
            f">>> SGD, learning_rate={learning_rate:0.5f}, weight_decay={weight_decay:0.5f}, momentum={momentum:0.3f}"
        )
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay, momentum=momentum)

    t0 = time.time()
    model_name = f"cellpose_{t0}" if model_name is None else model_name
    save_path = Path.cwd() if save_path is None else Path(save_path)
    model_path = save_path / "models" / model_name
    (save_path / "models").mkdir(exist_ok=True)

    train_logger.info(f">>> saving model to {model_path}")

    lavg, nsum = 0, 0
    for iepoch in range(n_epochs):
        np.random.seed(iepoch)
        if nimg != nimg_per_epoch:
            rperm = np.random.choice(np.arange(0, nimg), size=(nimg_per_epoch,),
                                     p=train_probs)
        else:
            rperm = np.random.permutation(np.arange(0, nimg))
        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[iepoch]
        net.train()
        # for k in range(0, nimg_per_epoch, batch_size):
        for imgi, lbl, diams in dataloader:
            rsc = diams / net.diam_mean.item() if rescale else np.ones(
                len(diams), "float32")
            # augmentations -> transform inside ZarrDataset
            imgi, lbl = transforms.random_rotate_and_resize(imgs, Y=lbls, rescale=rsc,
                                                            scale_range=scale_range,
                                                            xy=(bsize, bsize))[:2]

            X = torch.from_numpy(imgi).to(device)
            y = net(X)[0]
            loss = _loss_fn_seg(lbl, y, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            train_loss *= len(imgi)
            lavg += train_loss
            nsum += len(imgi)

        if iepoch == 5 or iepoch % 10 == 0:
            lavgt = 0.
            if test_data is not None or test_files is not None:
                np.random.seed(42)
                if nimg_test != nimg_test_per_epoch:
                    rperm = np.random.choice(np.arange(0, nimg_test),
                                             size=(nimg_test_per_epoch,), p=test_probs)
                else:
                    rperm = np.random.permutation(np.arange(0, nimg_test))
                for ibatch in range(0, len(rperm), batch_size):
                    with torch.no_grad():
                        net.eval()
                        inds = rperm[ibatch:ibatch + batch_size]
                        imgs, lbls = _get_batch(inds, data=test_data,
                                                labels=test_labels, files=test_files,
                                                labels_files=test_labels_files,
                                                **kwargs)
                        diams = np.array([diam_test[i] for i in inds])
                        rsc = diams / net.diam_mean.item() if rescale else np.ones(
                            len(diams), "float32")
                        imgi, lbl = transforms.random_rotate_and_resize(
                            imgs, Y=lbls, rescale=rsc, scale_range=scale_range,
                            xy=(bsize, bsize))[:2]
                        X = torch.from_numpy(imgi).to(device)
                        y = net(X)[0]
                        loss = train._loss_fn_seg(lbl, y, device)
                        test_loss = loss.item()
                        test_loss *= len(imgi)
                        lavgt += test_loss
                lavgt /= len(rperm)
            lavg /= nsum
            train_logger.info(
                f"{iepoch}, train_loss={lavg:.4f}, test_loss={lavgt:.4f}, LR={LR[iepoch]:.4f}, time {time.time()-t0:.2f}s"
            )
            lavg, nsum = 0, 0

        if iepoch > 0 and iepoch % save_every == 0:
            net.save_model(model_path)
    net.save_model(model_path)

    return model_path