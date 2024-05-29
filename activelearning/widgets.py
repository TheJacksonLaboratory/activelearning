from pathlib import Path
import torch
import napari
from napari.layers import Layer, Image, Labels
from napari.plugins.utils import get_potential_readers
import numpy as np
import zarr
import zarrdataset as zds
import dask.array as da

import datautils

from magicgui import magicgui
from cellpose import models, transforms


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

    u_sp_lab = np.zeros_like(super_pixel_labels, dtype=np.float32)

    for sp_l in super_pixel_indices:
        mask = super_pixel_labels == sp_l
        u_val = np.sum(mutual_info[mask]) / np.sum(mask)
        u_sp_lab = np.where(mask, u_val, u_sp_lab)

    return u_sp_lab


def cellpose_model_init():
    model = models.CellposeModel(gpu=False, model_type="cyto")

    model.net.load_model(model.pretrained_model[0], device=model.device)

    model.net.mkldnn = False
    model.net.eval()

    add_dropout(model.net)

    return model.net


def cellpose_inference(img, model):
    x = transforms.convert_image(img, None, normalize=False, invert=False,
                                 nchan=img.shape[-1])
    x = transforms.normalize_img(x, invert=False)

    x = torch.from_numpy(np.moveaxis(x, -1, 0))[None, :2, ...]

    with torch.no_grad():
        y, _ = model(torch.cat((x, x), dim=1))
        cellprob = y[0, 2, ...].detach().cpu()
        probs = cellprob.sigmoid().numpy()

    return probs


def save_zarr(output_filename, data, shape, chunk_size, group_name, dtype):
    out_grp = zarr.open(output_filename, mode="a")

    if isinstance(chunk_size, int):
        chunk_size = [chunk_size] * len(shape)

    chunks_size_axes = list(map(min, shape, chunk_size))

    out_grp.create_dataset(
        data=data,
        name=group_name,
        shape=shape,
        chunks=chunks_size_axes,
        compressor=zarr.Blosc(clevel=9),
        write_empty_chunks=False,
        dtype=dtype,
        overwrite=True
    )


@magicgui(
    patch_size={
        "widget_type": "SpinBox",
        "min": 128,
        "max": 4096,
        "step": 128,
        "label": "Patch size"
    }
)
def generate_blank_mask(image_layer: Image,
                        patch_size: int = 256,
                        ) -> Labels:
    """Generate a sampling mask for the input image.

    Parameters
    ----------
    image_layer: "napari.layers.Image"
        The image used to sample patches.
    scale: float
        The scale of the sampling mask.

    Returns
    -------
    mask_layer:
        A blank labels layer to draw the sampleable regions of the image.
    """
    source_axes = image_layer.metadata["source_axes"]
    image_shape = image_layer.data.shape

    mask_axes = "".join([ax for ax in source_axes if ax in "ZYX"])

    mask_shape = [max(1, s // patch_size)
                  for s, ax in zip(image_shape, source_axes)
                  if ax in mask_axes]

    mask_scale = tuple(
        patch_size if s // patch_size > 1 else 1
        for s, ax in zip(image_shape, source_axes)
        if ax in mask_axes
    )

    mask_translate = tuple(
        (scl - 1) / 2
        for scl in mask_scale
    )

    mask_layer = Labels(data=np.zeros(mask_shape, dtype=np.uint8),
                        scale=mask_scale,
                        translate=mask_translate,
                        metadata={"source_axes": mask_axes,
                                  "input_layer_name": image_layer.name},
                        name=image_layer.name + " sample-mask")

    return mask_layer


@magicgui(
    patch_size={
        "widget_type": "SpinBox",
        "min": 128,
        "max": 1024,
        "step": 128,
        "label": "Patch size"
    },
    max_samples={
        "widget_type": "SpinBox",
        "min": 1,
        "max": 2**32-1,
        "step": 10,
        "label": "Maximum samples"
    },
    MC_repetitions={
        "widget_type": "SpinBox",
        "min": 1,
        "max": 100,
        "step": 10,
        "label": "Monte Carlo repetitions"
    },
    output_dir={
        "widget_type": "FileEdit",
        "mode": "d"
    }
)
def compute_acquisition_function(image_layer: Image,
                                 mask_layer: Labels,
                                 patch_size: int = 256,
                                 max_samples: int = 100,
                                 MC_repetitions: int = 30,
                                 output_dir: str = "."
                                 ) -> Image:
    """Execute the computation of the acquisition function on the selected
    sampleable regions of the images.

    Parameters:
    -----------
    image_layer: napari.layers.Image
        Image used as input for feeding the segmentation model.
    mask_layer: napari.layers.Labels
        Mask used to determine the sampleable regions of the image.
    patch_size: int
        Size of the patch fed to the segmentation model.
    max_samples: int
        Maximum number of random patches extracted from sampelable regions.
    MC_repetitions: int
        Monte Carlo repetitions used to estimate the confidence of the
        segmentation model prediction.
    output_dir: str
        Directory where the sampling mask (if any) and the acquisition
        functions will be stored.

    Returns:
    --------
    acquisition_fun_layer: napari.layers.Image
        Acquisition function computed on samplable regions of the image
    """
    output_name = image_layer.name

    # Remove any invalid character from the name
    for chr in [" ", ".", "/", "\\"]:
        output_name = "-".join(output_name.split(chr))

    if image_layer.metadata["splitted_channels"]:
        displayed_source_axes = [
            ax
            for ax in image_layer.metadata["source_axes"]
            if ax != "C"
        ]

    else:
        displayed_source_axes = image_layer.metadata["source_axes"]

    acquisition_fun_shape = [
        s
        for s, ax in zip(image_layer.data.shape, displayed_source_axes)
        if ax in "ZYX" and s > 1
    ]

    acquisition_fun_chunk_size = [max(patch_size, 4096)]
    acquisition_fun_chunk_size *= len(acquisition_fun_shape)

    dataset_metadata = {
        "images": zds.ImagesDatasetSpecs(
            filenames=image_layer.metadata["filenames"],
            data_group=image_layer.metadata["data_group"],
            source_axes=image_layer.metadata["source_axes"],
            axes=image_layer.metadata["axes"],
            roi=image_layer.metadata["roi"],
        ),
    }

    output_name = image_layer.name

    # Remove any invalid character from the name
    for chr in [" ", ".", "/", "\\"]:
        output_name = "-".join(output_name.split(chr))

    output_filename = output_dir / Path(output_name + ".zarr")

    if mask_layer:
        mask_axes = "".join([
            ax
            for ax in displayed_source_axes
            if ax in "ZYX"
        ])

        save_zarr(output_filename, data=mask_layer.data,
                  shape=mask_layer.data.shape,
                  chunk_size=max(1, int(patch_size ** 0.5)),
                  group_name="sampling_mask",
                  dtype=bool)

        dataset_metadata["masks"] = zds.MasksDatasetSpecs(
            filenames=str(output_filename),
            data_group="sampling_mask",
            source_axes=mask_axes,
            axes=mask_axes
        )

    save_zarr(output_filename, data=None, shape=acquisition_fun_shape,
              chunk_size=acquisition_fun_chunk_size,
              group_name="acquisition_fun/0",
              dtype=np.float32)

    acquisition_fun = zarr.open(str(output_filename) + "/acquisition_fun/0",
                                mode="a")

    dl = datautils.get_dataloader(dataset_metadata, patch_size=patch_size,
                                  shuffle=True)

    model = cellpose_model_init()

    acquisition_fun_max = 0
    n_samples = 0
    for pos, img, img_sp in dl:
        probs = []
        for _ in range(MC_repetitions):
            probs.append(
                cellpose_inference(img[0].numpy(), model)
            )

        probs = np.stack(probs, axis=0)

        u_sp_lab = compute_acquisition(probs, img_sp[0].numpy())

        pos_u_lab = (slice(pos[0, 0, 0].item(), pos[0, 0, 1].item()),
                     slice(pos[0, 1, 0].item(), pos[0, 1, 1].item()))

        acquisition_fun[pos_u_lab] = u_sp_lab
        acquisition_fun_max = max(acquisition_fun_max, u_sp_lab.max())

        n_samples += 1

        if n_samples >= max_samples:
            break

    # Downsample the acquisition function
    acquisition_fun = da.from_zarr(output_filename,
                                   component="acquisition_fun/0")

    max_downsampling_levels = int(np.log(min(acquisition_fun.shape))
                                  / np.log(4))

    acquisition_fun_multiscale = [
        acquisition_fun[::4 ** level, ::4 ** level]
        for level in range(max_downsampling_levels)
    ]

    acquisition_fun_layer = Image(
        data=acquisition_fun_multiscale,
        multiscale=True,
        name=image_layer.name + " acquisition function",
        opacity=0.8,
        blending="translucent_no_depth",
        colormap="magma",
        contrast_limits=(0, acquisition_fun_max)
    )

    return acquisition_fun_layer


@magicgui(
    is_splitted={
        "widget_type": "CheckBox",
        "text": "Are channels splitted into layers?"
    }
)
def metadata_manager(viewer: napari.Viewer,
                     layer: Layer,
                     source_axes: str = "TCZYX",
                     is_splitted: bool = False) -> None:
    """Manage a set of images for visualization.

    Parameters:
    -----------
    viewer: napari.Viewer
        Current viewer used to determine the axes used as inputs.
    layer: napari.layers.Layer
        An existing layer in the viewer.
    source_axes: str
        The order of the input image axes.
    is_splitted: bool
        Whether the channels of the image are splitted into different layers or
        they are in the same layer.

    Returns:
    --------
    image_layer: napari.layer.Image
        Acquisition function computed on samplable regions of the image
    """
    input_filename = layer._source.path
    data_group = ""

    if input_filename:
        input_filename = str(input_filename)
        data_group = "/".join(input_filename.split(".")[-1].split("/")[1:])

    if data_group:
        data_group = data_group + "/0"
        input_filename = input_filename[:-len(data_group) - 1]
    else:
        data_group = None

    # Generate the current ROI for images with non-channel, non-spatial axes
    if is_splitted:
        displayed_source_axes = [ax for ax in source_axes if ax != "C"]
    else:
        displayed_source_axes = source_axes

    roi_start = [0] * len(displayed_source_axes)
    roi_length = [-1] * len(displayed_source_axes)

    for ord in viewer.dims.order[:-viewer.dims.ndisplay]:
        roi_start[ord] = int(viewer.cursor.position[ord])
        roi_length[ord] = 1

    if is_splitted and "C" in source_axes:
        roi_start.insert(source_axes.index("C"), 0)
        roi_length.insert(source_axes.index("C"), -1)

    axes = ["Y", "X", "C"]
    if "C" not in source_axes:
        axes.remove("C")
    axes = "".join(axes)

    if input_filename:
        source_data = input_filename
    else:
        source_data = layer.data

    layer.scale = tuple([1] * len(layer.scale))

    layer.metadata["splitted_channels"] = is_splitted
    layer.metadata["filenames"] = source_data
    layer.metadata["data_group"] = data_group
    layer.metadata["source_axes"] = source_axes
    layer.metadata["axes"] = axes
    layer.metadata["roi"] = [",".join(map(str, roi_start))
                             + ":"
                             + ",".join(map(str, roi_length))]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Active learning sample generator")
    parser.add_argument("-ps", "--patch-size", dest="patch_size", type=int,
                        help="Size of the sampled patches",
                        default=256)
    parser.add_argument("-ns", "--num-samples", dest="samples_per_image",
                        type=int,
                        help="Maximum number of samples to extract from each "
                             "image")
    parser.add_argument("-ms", "--max-samples", dest="max_samples_to_annotate",
                        type=int,
                        help="Maximum number of samples to annotate after all"
                             " samples have been extracted")
    parser.add_argument("-t", "--repetitions", dest="repetitions", type=int,
                        help="Number of repetitions for Monte Carlo sampling",
                        default=30)
    parser.add_argument("-w", "--num-workers", dest="num_workers", type=int,
                        help="Number of workers for multi thread data loading",
                        default=0)

    args = parser.parse_args()

    viewer = napari.Viewer()

    viewer.window.add_dock_widget(metadata_manager)
    viewer.window.add_dock_widget(generate_blank_mask)
    viewer.window.add_dock_widget(compute_acquisition_function)

    napari.run()
