import pytest

import os

import numpy as np
import zarr
import dask.array as da

from napari.layers import Layer
from napari_activelearning._utils import (get_source_data, downsample_image,
                                          save_zarr,
                                          validate_name)


def test_get_source_data():
    layer = Layer(data=[1, 2, 3])
    input_filename, data_group = get_source_data(layer)
    assert input_filename is None
    assert data_group is None

    layer = Layer(data=[1, 2, 3], metadata={'path': '/path/to/data.zarr', 'data_group': 'group'})
    input_filename, data_group = get_source_data(layer)
    assert input_filename == '/path/to/data.zarr'
    assert data_group == 'group'


def test_downsample_image():
    # Create a test input array
    input_array = np.random.rand(100, 100)
    input_dask_array = da.from_array(input_array)

    # Create a test Zarr store
    zarr_store = zarr.MemoryStore()
    zarr_group = zarr.group(store=zarr_store)

    # Save the input array to the Zarr store
    zarr_group.create_dataset('data', data=input_dask_array)

    # Call the downsample_image function
    downsampled_zarr = downsample_image(zarr_group, ['Y', 'X'], 'data', scale=2, num_scales=3)

    # Check the shape of the downsampled array
    assert downsampled_zarr[0].shape == (100, 100)
    assert downsampled_zarr[1].shape == (50, 50)
    assert downsampled_zarr[2].shape == (25, 25)


def test_save_zarr():
    # Create a test output filename
    output_filename = "test_output.zarr"

    # Create a test data array
    data = np.random.rand(100, 100)

    # Create test parameters
    shape = data.shape
    chunk_size = 10
    name = "test_data"
    dtype = data.dtype

    # Call the save_zarr function
    save_zarr(output_filename, data, shape, chunk_size, name, dtype)

    # Check if the output file exists
    assert os.path.exists(output_filename)

    # Check if the saved data matches the original data
    saved_data = zarr.open(output_filename, mode="r")[name][:]
    assert np.array_equal(saved_data, data)

    # Clean up the test output file
    os.remove(output_filename)


def test_validate_name():
    group_names = {"Group1", "Group2", "Group3"}

    # Test case 1: New child name is not in group names
    previous_child_name = "Group1"
    new_child_name = "Group4"
    expected_result = "Group4"
    assert validate_name(group_names, previous_child_name, new_child_name) == expected_result

    # Test case 2: New child name is already in group names
    previous_child_name = "Group1"
    new_child_name = "Group2"
    expected_result = "Group2 (1)"
    assert validate_name(group_names, previous_child_name, new_child_name) == expected_result

    # Test case 3: New child name is empty
    previous_child_name = "Group1"
    new_child_name = ""
    expected_result = "Group1"
    assert validate_name(group_names, previous_child_name, new_child_name) == expected_result

    # Test case 4: Previous child name is not in group names
    previous_child_name = "Group4"
    new_child_name = "Group5"
    expected_result = "Group5"
    assert validate_name(group_names, previous_child_name, new_child_name) == expected_result

    # Test case 5: Previous child name is empty
    previous_child_name = ""
    new_child_name = "Group6"
    expected_result = "Group6"
    assert validate_name(group_names, previous_child_name, new_child_name) == expected_result