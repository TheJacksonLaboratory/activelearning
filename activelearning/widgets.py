from typing import Tuple
import os
import uuid
import napari
from napari.layers import Image, Labels
import numpy as np
import zarr
import dask.array as da
from magicgui import magicgui


def save_zarr(output_filename, data, shape, chunk_size, group_name, dtype):
    if output_filename is not None and len(output_filename):
        out_grp = zarr.open(output_filename, mode="a")
    else:
        out_grp = zarr.open(zarr.MemoryStore())

    if isinstance(chunk_size, int):
        chunk_size = [chunk_size] * len(chunk_size)

    chunks_size_axes = list(map(min, shape, chunk_size))

    new_array = out_grp.create_dataset(
        data=data,
        name=group_name,
        shape=shape,
        chunks=chunks_size_axes,
        compressor=zarr.Blosc(clevel=9),
        write_empty_chunks=False,
        dtype=dtype,
        overwrite=True
    )

    return new_array


@magicgui
def save_all_masks(viewer: napari.Viewer, output_dir: str = "./",
                   chunk_size: int = 16):
    """Save all masks as zarr files for using in following steps.

    Parameters:
    -----------
    viewer: napari.Viewer
        Current napari viewer.
    output_dir: str
        Output directory where all masks are saved.
    chunk_size: int
        Size of the chunks used to store the mask arrays.

    Returns:
    --------
    Nothing
    """
    for layer in viewer.layers:
        if isinstance(layer, Labels):
            output_name = layer.metadata.get("input_layer_name",
                                             "input-" + str(uuid.uuid1()))

            # Remove any invalid character from the name
            for chr in [" ", ".", "/", "\\"]:
                output_name = "-".join(output_name.split(chr))

            output_filename = os.path.join(output_dir,
                                           output_name + ".zarr")

            save_zarr(output_filename, data=layer.data, shape=layer.data.shape,
                      chunk_size=chunk_size,
                      group_name="sampling_mask",
                      dtype=bool)


@magicgui(
        patch_size={
            "widget_type": "SpinBox",
            "min": 128,
            "max": 1024,
            "step": 128,
            "label": "Patch size"
        }
)
def generate_blank_mask(image_layer: Image,
                        axes_order: str = "TCZYX",
                        spatial_axes: str = "ZYX",
                        patch_size: int = 256
                        ) -> Labels:
    """Generate a sampling mask for the input image.

    Parameters
    ----------
    image_layer: "napari.layers.Image"
        The image used to sample patches.
    axes_order: str
        The order of the input image axes.
    spatial_axes: str
        The maskable axes of the image.
    scale: float
        The scale of the sampling mask.

    Returns
    -------
    sampling_mask:
        A blank labels layer to draw the sampleable regions of the image.
    """
    mask_axes = "".join([ax for ax in axes_order if ax in spatial_axes])

    mask_shape = [
        max(1, s // patch_size)
        for s, ax in zip(image_layer.data.shape, axes_order)
        if ax in mask_axes
    ]

    mask_scale = tuple(
        patch_size if s // patch_size > 1 else 1
        for s, ax in zip(image_layer.data.shape, axes_order)
        if ax in mask_axes
    )

    mask_translate = tuple(
        (scl - 1) / 2
        for scl in mask_scale
    )

    sampling_mask = Labels(data=np.zeros(mask_shape, dtype=np.uint8),
                           scale=mask_scale,
                           translate=mask_translate,
                           metadata={"source_axes": mask_axes,
                                     "input_layer_name": image_layer.name},
                           name=image_layer.name + " sample-mask")

    return sampling_mask


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
)
def compute_acquisition_function(image_layer: Image,
                                 mask_layer: Labels,
                                 axes_order: str = "TCZYX",
                                 patch_size: int = 256,
                                 max_samples: int = 100,
                                 MC_repetitions: int = 30,
                                 output_dir: str = "None",
                                 ) -> Image:
    """Execute the computation of the acquisition function on the selected
    sampleable regions of the images.

    Parameters:
    -----------
    image_layer: napari.layers.Image
        Image used as input for feeding the segmentation model.
    axes_order: str
        The order of the input image axes.
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
        If passed, the confidence map is stored in that directory.

    Returns:
    --------
    acquisition_fun_layer: napari.layers.Image
        Acquisition function computed on samplable regions of the image
    """
    output_name = image_layer.name

    # Remove any invalid character from the name
    for chr in [" ", ".", "/", "\\"]:
        output_name = "-".join(output_name.split(chr))

    acquisition_fun_shape = [
        s
        for s, ax in zip(image_layer.data.shape, axes_order)
        if ax in "YXZ" and s > 1
    ]

    acquisition_fun_chunk_size = [max(patch_size, 4096)]
    acquisition_fun_chunk_size *= len(acquisition_fun_shape)

    if (output_dir is not None and len(output_dir)
       and output_dir.strip(" ").lower() != "none"):
        output_filename = os.path.join(output_dir,
                                       output_name + ".zarr")
    else:
        output_filename = None

    acquisition_fun = save_zarr(output_filename, data=None,
                                shape=acquisition_fun_shape,
                                chunk_size=acquisition_fun_chunk_size,
                                group_name="acqusition_fun/0",
                                dtype=np.float32)
    
    acquisition_fun_max = 1000
    # TODO: Compute acquisition function
    
    # Downsample the acquisition function
    acquisition_fun = da.from_zarr(acquisition_fun)

    max_downsampling_levels = int(np.log(min(acquisition_fun_shape))
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

    viewer.window.add_dock_widget(generate_blank_mask)
    viewer.window.add_dock_widget(save_all_masks)
    viewer.window.add_dock_widget(compute_acquisition_function)

    napari.run()
