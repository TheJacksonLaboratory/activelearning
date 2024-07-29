import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
import zarr

from napari_activelearning._acquisition import (AcquisitionFunction,
                                                add_multiscale_output_layer)
from napari_activelearning._layers import LayerChannel, ImageGroup, LayersGroup
from napari_activelearning._models import SimpleTunable

try:
    import torch
    USING_PYTORCH = True
except ModuleNotFoundError:
    USING_PYTORCH = False


@pytest.fixture
def image_groups_manager():
    return MagicMock()


@pytest.fixture
def labels_manager():
    return MagicMock()


@pytest.fixture
def layers_group():
    layer = MagicMock()
    layer.name = "Image layer"
    layers_group_mock = MagicMock()
    layers_group_mock.add_layer.return_value = LayerChannel(layer)
    return layers_group_mock


@pytest.fixture
def image_group(layers_group):
    image_group_mock = MagicMock()
    image_group_mock.getLayersGroup.return_value = layers_group
    return image_group_mock


@pytest.fixture
def tunable_segmentation_method():
    method = MagicMock()
    method.probs.return_value = np.random.random((10, 10))
    method.segment.return_value = np.random.randint(0, 2, (10, 10))
    return method


@pytest.fixture
def acquisition_function(image_groups_manager, labels_manager,
                         tunable_segmentation_method):
    with patch('napari.current_viewer') as mock_viewer:
        mock_viewer.return_value.dims.axis_labels = ['t', 'z', 'y', 'x']
        return AcquisitionFunction(image_groups_manager, labels_manager,
                                   tunable_segmentation_method)


@pytest.fixture(scope="module", params=[True, False])
def output_dir(request, tmpdir_factory):
    if request.param:
        tmp_dir = tmpdir_factory.mktemp("output")
        tmp_dir_path = Path(tmp_dir)
    else:
        tmp_dir_path = None

    yield tmp_dir_path


@pytest.fixture
def root_array():
    root_array_data = zarr.empty((1, 3, 1, 10, 10))
    yield root_array_data

@pytest.fixture
def output_array(root_array, output_dir):
    output_filename = None

    if output_dir is not None:
        output_filename = output_dir / "out.zarr"
        z_group = zarr.open(output_filename)
    else:
        z_group = zarr.open()

    z_group.create_dataset("data", data=root_array)

    yield (z_group, output_filename)


def test_compute_acquisition_fun(acquisition_function,
                                 tunable_segmentation_method):
    img = np.random.random((10, 10, 3))
    img_sp = np.random.random((10, 10))
    MC_repetitions = 3
    result = acquisition_function._compute_acquisition_fun(img, img_sp,
                                                           MC_repetitions)
    assert result is not None
    assert tunable_segmentation_method.probs.call_count == MC_repetitions


def test_compute_segmentation(acquisition_function,
                              tunable_segmentation_method):
    img = np.random.random((1, 1, 1, 10, 10, 3))
    labels_offset = 1
    result = acquisition_function._compute_segmentation(img, labels_offset)
    expected_segmentation = tunable_segmentation_method.segment()
    expected_segmentation = np.where(expected_segmentation,
                                     expected_segmentation + labels_offset,
                                     expected_segmentation)
    assert np.array_equal(result, expected_segmentation)


def test_compute_acquisition(acquisition_function):
    dataset_metadata = {
        "images": {"source_axes": "TCZYX", "axes": "TZYXC"},
        "masks": {"source_axes": "TZYX", "axes": "TZYX"}
    }
    acquisition_fun = np.zeros((1, 1, 10, 10))
    segmentation_out = np.zeros((1, 1, 10, 10))
    sampling_positions = None
    segmentation_only = False

    acquisition_function.input_axes = "TZYX"
    acquisition_function.model_axes = "YXC"
    acquisition_function.patch_sizes = {"T": 1, "Z": 1, "Y": 10, "X": 10}

    with (patch('napari_activelearning._acquisition.get_dataloader')
          as mock_dataloader):
        if USING_PYTORCH:
            mock_dataloader.return_value = [
                (torch.LongTensor([[[0, 1], [0, 1], [0, 10], [0, 10],
                                    [0, -1]]]),
                 torch.zeros((1, 1, 1, 10, 10, 3)),
                 torch.zeros((1, 1, 1, 10, 10, 1)))
            ]
        else:
            mock_dataloader.return_value = [
                (np.array([[0, 1], [0, 1], [0, 10], [0, 10], [0, -1]]),
                 np.zeros((1, 1, 10, 10, 3)),
                 np.zeros((1, 1, 10, 10, 1)))
            ]
        result = acquisition_function.compute_acquisition(
            dataset_metadata, acquisition_fun, segmentation_out,
            sampling_positions,
            segmentation_only
        )

        assert len(result) == 1


def test_add_multiscale_output_layer(output_array, image_group,
                                     make_napari_viewer):
    root_array, output_filename = output_array
    axes = "TCZYX"
    scale = [1, 1, 1, 1, 1]
    data_group = "data"
    group_name = "group"
    layers_group_name = "layers_group"
    reference_source_axes = "TCZYX"
    reference_scale = [1, 1, 1, 1, 1]
    contrast_limits = [0, 1]
    colormap = "gray"
    viewer = make_napari_viewer()
    add_func = viewer.add_image

    output_channel = add_multiscale_output_layer(
        root_array,
        axes,
        scale,
        data_group,
        group_name,
        layers_group_name,
        image_group,
        reference_source_axes,
        reference_scale,
        output_filename,
        contrast_limits,
        colormap,
        add_func
    )

    assert isinstance(output_channel, LayerChannel)


def test_prepare_datasets_metadata(acquisition_function):
    # Define the input parameters for the method
    image_group = ImageGroup()
    output_axes = "TCZYX"
    displayed_source_axes = "TCZYX"
    displayed_shape = [1, 3, 10, 10, 10]
    layer_types = [(LayersGroup(), "images")]

    # Call the method
    dataset_metadata, sampling_positions = acquisition_function._prepare_datasets_metadata(
        image_group,
        output_axes,
        displayed_source_axes,
        displayed_shape,
        layer_types
    )

    # Assert the output values
    assert dataset_metadata == {
        "images": {
            "filenames": None,
            "data_group": None,
            "source_axes": "TCZYX",
            "axes": "TCZYX",
            "roi": [[slice(None), slice(None), slice(None), slice(None), slice(None)]],
            "modality": "images"
        }
    }
    assert sampling_positions is None


def test_compute_acquisition_layers(acquisition_function):
    # Mock the necessary dependencies and setup the test data
    image_group = MagicMock()
    acquisition_function.image_groups_manager.groups_root.childCount.return_value = 1
    acquisition_function.image_groups_manager.groups_root.child.return_value = image_group
    acquisition_function.image_groups_manager.groups_root.child.return_value.getSelected.return_value = True
    acquisition_function.image_groups_manager.groups_root.child.return_value.group_name = "test_group"
    acquisition_function.image_groups_manager.groups_root.child.return_value.group_dir = "/path/to/group"
    acquisition_function.image_groups_manager.groups_root.child.return_value.input_layers_group = 0
    acquisition_function.image_groups_manager.groups_root.child.return_value.child.return_value.source_axes = "TCZYX"
    acquisition_function.image_groups_manager.groups_root.child.return_value.child.return_value.shape = (1, 3, 10, 10, 10)
    acquisition_function.image_groups_manager.groups_root.child.return_value.child.return_value.scale = (1.0, 1.0, 1.0, 1.0, 1.0)
    acquisition_function.image_groups_manager.groups_root.child.return_value.sampling_mask_layers_group = None
    acquisition_function._prepare_datasets_metadata.return_value = ({}, None)

    # Call the method under test
    acquisition_function.compute_acquisition_layers(run_all=True, segmentation_group_name="segmentation", segmentation_only=False)

    # Assert that the necessary methods were called with the expected arguments
    acquisition_function.image_groups_manager.groups_root.childCount.assert_called_once()
    acquisition_function.image_groups_manager.groups_root.child.assert_called_once_with(0)
    acquisition_function.image_groups_manager.groups_root.child.return_value.getSelected.assert_called_once()
    acquisition_function.image_groups_manager.groups_root.child.return_value.group_name.assert_called_once()
    acquisition_function.image_groups_manager.groups_root.child.return_value.group_dir.assert_called_once()
    acquisition_function.image_groups_manager.groups_root.child.return_value.input_layers_group.assert_called_once()
    acquisition_function.image_groups_manager.groups_root.child.return_value.child.assert_called_once_with(0)
    acquisition_function.image_groups_manager.groups_root.child.return_value.child.return_value.source_axes.assert_called_once()
    acquisition_function.image_groups_manager.groups_root.child.return_value.child.return_value.shape.assert_called_once()
    acquisition_function.image_groups_manager.groups_root.child.return_value.child.return_value.scale.assert_called_once()
    acquisition_function.image_groups_manager.groups_root.child.return_value.sampling_mask_layers_group.assert_called_once()
    acquisition_function._prepare_datasets_metadata.assert_called_once_with(
        image_group,
        "TCZYX",
        "TCZYX",
        (1, 3, 10, 10, 10),
        [(acquisition_function.image_groups_manager.groups_root.child.return_value.child.return_value, "images"),
         (acquisition_function.image_groups_manager.groups_root.child.return_value.sampling_mask_layers_group, "masks")]
    )


@pytest.fixture
def tunable_method():
    return SimpleTunable()


def test_fine_tune(tunable_method):
    dataset_metadata_list = [
        (
            {
                "images": {
                    "filenames": ["image1.tif", "image2.tif"],
                    "data_group": "data",
                    "source_axes": "YXC",
                    "axes": "YXC",
                    "roi": None,
                    "modality": "images"
                }
            },
            [
                [1, 2, 3],
                [4, 5, 6]
            ]
        ),
        (
            {
                "images": {
                    "filenames": ["image3.tif", "image4.tif"],
                    "data_group": "data",
                    "source_axes": "YXC",
                    "axes": "YXC",
                    "roi": None,
                    "modality": "images"
                }
            },
            [
                [7, 8, 9],
                [10, 11, 12]
            ]
        )
    ]

    train_data_proportion = 0.8
    patch_sizes = 256
    model_axes = "YXC"

    tunable_method.fine_tune(dataset_metadata_list, train_data_proportion, patch_sizes, model_axes)

    # Add assertions here to verify the behavior of the fine_tune method
    assert tunable_method.train_data_proportion == train_data_proportion
    assert tunable_method.patch_sizes == patch_sizes
    assert tunable_method.model_axes == model_axes
