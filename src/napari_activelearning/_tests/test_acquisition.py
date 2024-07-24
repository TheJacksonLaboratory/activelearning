import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from napari_activelearning._acquisition import AcquisitionFunction


@pytest.fixture
def image_groups_manager():
    return MagicMock()


@pytest.fixture
def labels_manager():
    return MagicMock()


@pytest.fixture
def tunable_segmentation_method():
    method = MagicMock()
    method.probs.return_value = np.random.random((10, 10, 10))
    method.segment.return_value = np.random.randint(0, 2, (10, 10, 10))
    return method


@pytest.fixture
def acquisition_function(image_groups_manager, labels_manager,
                         tunable_segmentation_method):
    with patch('napari.current_viewer') as mock_viewer:
        mock_viewer.return_value.dims.axis_labels = ['t', 'z', 'y', 'x']
        return AcquisitionFunction(image_groups_manager, labels_manager,
                                   tunable_segmentation_method)


def test_initialization(acquisition_function):
    assert acquisition_function._patch_size == 128
    assert acquisition_function._max_samples == 1
    assert acquisition_function._MC_repetitions == 3
    assert acquisition_function._input_axes == "TZYX"


def test_compute_acquisition_fun(acquisition_function,
                                 tunable_segmentation_method):
    img = np.random.random((10, 10, 1))
    img_sp = np.random.random((10, 10, 1))
    MC_repetitions = 3
    result = acquisition_function._compute_acquisition_fun(img, img_sp,
                                                           MC_repetitions)
    assert result is not None
    assert tunable_segmentation_method.probs.call_count == MC_repetitions


def test_compute_segmentation(acquisition_function,
                              tunable_segmentation_method):
    img = np.random.random((10, 10, 10))
    labels_offset = 1
    result = acquisition_function._compute_segmentation(img, labels_offset)
    expected_segmentation = tunable_segmentation_method.segment()
    expected_segmentation = np.where(expected_segmentation,
                                     expected_segmentation + labels_offset,
                                     expected_segmentation)
    assert np.array_equal(result, expected_segmentation)


def test_compute_acquisition(acquisition_function):
    dataset_metadata = {
        "images": {"source_axes": "TCZYX"},
        "masks": {"source_axes": "ZYX"}
    }
    acquisition_fun = np.zeros((10, 10))
    segmentation_out = np.zeros((10, 10))
    sampling_positions = None
    segmentation_only = False
    spatial_axes = "ZYX"
    input_axes = "YXC"

    with patch('napari.current_viewer') as mock_viewer:
        mock_viewer.return_value.dims.axis_labels = ['t', 'z', 'y', 'x']
        mock_viewer.return_value.dims.current_step = [0, 17, 20, 30]
        mock_viewer.return_value.dims.order = [0, 1, 2, 3]

    with (patch('napari_activelearning._acquisition.get_dataloader')
          as mock_dataloader):
        mock_dataloader.return_value = [(torch.zeros((1, 2, 3)),
                                         torch.zeros((1, 10, 10, 1)),
                                         torch.zeros((1, 10, 10, 1)))]
        result = acquisition_function.compute_acquisition(
            dataset_metadata, acquisition_fun, segmentation_out,
            sampling_positions,
            segmentation_only,
            spatial_axes,
            input_axes
        )

        assert len(result) == 1
