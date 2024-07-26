import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from napari_activelearning._acquisition import AcquisitionFunction

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
                 np.array((1, 1, 10, 10, 3)),
                 np.array((1, 1, 10, 10, 1)))
            ]
        result = acquisition_function.compute_acquisition(
            dataset_metadata, acquisition_fun, segmentation_out,
            sampling_positions,
            segmentation_only
        )

        assert len(result) == 1
