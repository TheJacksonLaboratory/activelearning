import shutil
from pathlib import Path
import operator

import numpy as np
import zarrdataset as zds
import zarr

from napari.layers._multiscale_data import MultiScaleData

from napari_activelearning._utils import (get_source_data, downsample_image,
                                          save_zarr,
                                          validate_name,
                                          get_basename,
                                          get_dataloader,
                                          StaticPatchSampler,
                                          SuperPixelGenerator)

try:
    import torch
    USING_PYTORCH = True
except ModuleNotFoundError:
    USING_PYTORCH = False


def test_get_source_data(sample_layer):
    layer, org_source_data, org_input_filename, org_data_group = sample_layer
    input_filename, data_group, available_data_groups = get_source_data(layer)

    assert (not isinstance(input_filename, (Path, str))
            or (Path(input_filename.lower())
                == Path(str(org_input_filename).lower())))
    assert (isinstance(input_filename, (Path, str))
            or (isinstance(input_filename, (MultiScaleData, list))
                and all(map(np.array_equal, input_filename, org_source_data)))
            or np.array_equal(input_filename, org_source_data))
    assert (not isinstance(input_filename, (Path, str))
            or (Path(str(data_group).lower())
                == Path(str(org_data_group).lower())))

    assert isinstance(available_data_groups, list)


def test_downsample_image(single_scale_type_variant_array):
    (source_data,
     input_filename,
     data_group,
     array_shape) = single_scale_type_variant_array

    downsample_scale = 2
    num_scales = 10
    if data_group and "/" in data_group:
        data_group_parts = Path(data_group).parts[0]
        if len(data_group_parts) == 1:
            data_group_root = data_group_parts[0]
        else:
            data_group_root = str(Path(*data_group_parts[:-1]))
    else:
        data_group_root = ""

    if input_filename is not None:
        source_data = input_filename

    downsampled_zarr = downsample_image(
        source_data,
        axes="TCZYX",
        scale={ax: 1 for ax in "TCZYX"},
        data_group=data_group,
        downsample_scale=downsample_scale,
        num_scales=num_scales,
        reference_source_axes="TCZYX",
        reference_scale={"T": 1, "C": 1, "Z": 1, "Y": 1, "X": 1},
        reference_units=None
    )

    if isinstance(array_shape, list):
        array_shape = array_shape[0]

    min_spatial_shape = min(array_shape["TCZYX".index(ax)] for ax in "ZYX")

    expected_scales = min(num_scales,
                          int(np.log(min_spatial_shape)
                              / np.log(downsample_scale)))

    expected_shapes = [
        [int(np.ceil(ax_s / (downsample_scale ** s))) if ax in "YX" else ax_s
         for ax, ax_s in zip("TCZYX", array_shape)
         ]
        for s in range(expected_scales)
    ]

    assert len(downsampled_zarr) == expected_scales
    assert all(map(lambda src_shape, dwn_arr:
                   all(map(operator.eq, src_shape, dwn_arr.shape)),
                   expected_shapes,
                   downsampled_zarr))

    if isinstance(input_filename, (Path, str)):
        z_root = zarr.open(input_filename, mode="r")
        assert all(map(lambda scl: str(scl) in z_root[data_group_root],
                       range(expected_scales)))
        assert "multiscales" in z_root[data_group_root].attrs

        for scl in range(1, expected_scales):
            shutil.rmtree(input_filename / data_group_root / str(scl))


def test_save_zarr(sample_layer, output_group):
    layer, source_data, input_filename, data_group = sample_layer
    name = "test_data"
    group_name = name

    is_multiscale = isinstance(layer.data, (MultiScaleData, list))

    out_grp, out_grp_data = save_zarr(
        output_group,
        layer.data,
        layer.data.shape,
        True,
        name,
        layer.data.dtype,
        is_multiscale=is_multiscale,
        metadata=None,
        is_label=True,
        overwrite=True
    )

    assert group_name in out_grp
    assert (not is_multiscale
            or len(out_grp[name]) == len(layer.data))
    assert (isinstance(out_grp.store, zarr.MemoryStore)
            or layer.data.dtype in (np.float32, np.float64)
            or (isinstance(out_grp.store, (zarr.MemoryStore,
                                           zarr.DirectoryStore))
                and "image-label" in out_grp[group_name].attrs)
            or (not isinstance(out_grp.store, (zarr.MemoryStore,
                                               zarr.DirectoryStore))
                and "image-label" not in out_grp[group_name].attrs))


def test_validate_name():
    group_names = {"Group1", "Group2", "Group3"}

    # Test case 1: New child name is not in group names
    previous_child_name = None
    new_child_name = "Group4"
    expected_result = "Group4"
    assert validate_name(group_names,
                         previous_child_name,
                         new_child_name) == expected_result

    # Test case 2: New child name is already in group names
    previous_child_name = "Group1"
    new_child_name = "Group2"
    expected_result = "Group2 (1)"
    assert validate_name(group_names,
                         previous_child_name,
                         new_child_name) == expected_result

    # Test case 3: New child name is empty
    previous_child_name = "Group2 (1)"
    new_child_name = ""
    expected_result = ""
    assert validate_name(group_names,
                         previous_child_name,
                         new_child_name) == expected_result

    # Test case 4: Previous child name is not in group names
    previous_child_name = "Group1"
    new_child_name = "Group5"
    expected_result = "Group5"
    assert validate_name(group_names,
                         previous_child_name,
                         new_child_name) == expected_result


def test_get_basename():
    layer_name = "sample_layer"
    expected_result = "sample_layer"
    assert get_basename(layer_name) == expected_result

    layer_name = "sample_layer 1"
    expected_result = "sample_layer"
    assert get_basename(layer_name) == expected_result


def test_get_dataloader(dataset_metadata):
    patch_size = {"Y": 64, "X": 64}
    shuffle = True
    num_workers = 4
    batch_size = 8
    spatial_axes = "YX"
    model_input_axes = "YXC"

    dataloader = get_dataloader(
        dataset_metadata,
        patch_size=patch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
        spatial_axes=spatial_axes,
        model_input_axes=model_input_axes,
        tunable_method=None
    )

    if USING_PYTORCH:
        assert isinstance(dataloader.dataset._patch_sampler, zds.PatchSampler)
    else:
        assert isinstance(dataloader._patch_sampler, zds.PatchSampler)


def test_compute_chunks(image_collection):
    patch_size = {"Z": 1, "Y": 5, "X": 5}
    top_lefts = [[3, 0, 0], [3, 0, 5], [3, 5, 0], [3, 5, 5]]

    patch_sampler = StaticPatchSampler(patch_size=patch_size,
                                       top_lefts=top_lefts)

    expected_output = [dict(X=slice(0, 10), Y=slice(0, 10), Z=slice(0, 10))]

    chunks_slices = patch_sampler.compute_chunks(image_collection)

    assert chunks_slices == expected_output


def test_compute_patches(image_collection):
    patch_size = {"Z": 1, "Y": 5, "X": 5}
    top_lefts = [[3, 0, 0], [3, 0, 5], [3, 5, 0], [3, 5, 5]]
    chunk_tl = dict(X=slice(None), Y=slice(None), Z=slice(None))

    patch_sampler = StaticPatchSampler(patch_size=patch_size,
                                       top_lefts=top_lefts)

    chunks_slices = patch_sampler.compute_patches(image_collection, chunk_tl)

    # Assert that the number of chunks is equal to the number of top_lefts
    assert len(chunks_slices) == len(top_lefts)

    # Assert that each chunk slice has the correct shape
    for chunk_slices in chunks_slices:
        assert (chunk_slices["Z"].stop - chunk_slices["Z"].start
                == patch_size["Z"])
        assert (chunk_slices["Y"].stop - chunk_slices["Y"].start
                == patch_size["Y"])
        assert (chunk_slices["X"].stop - chunk_slices["X"].start
                == patch_size["X"])


def test_compute_transform():
    generator = SuperPixelGenerator(num_superpixels=25, axes="YXC",
                                    model_axes="YXC")
    image = np.random.random((10, 10, 3))
    labels = generator._compute_transform(image)
    assert labels.shape == (10, 10, 1)
    assert np.unique(labels).size == 25
