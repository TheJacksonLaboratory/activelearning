from unittest.mock import patch
import numpy as np

from napari_activelearning._acquisition import (AcquisitionFunction,
                                                compute_acquisition_fun,
                                                compute_segmentation,
                                                add_multiscale_output_layer)
from napari_activelearning._layers import LayerChannel

try:
    import torch
    USING_PYTORCH = True
except ModuleNotFoundError:
    USING_PYTORCH = False


def test_compute_acquisition_fun(tunable_segmentation_method):
    img = np.random.random((10, 10, 3))
    img_sp = np.random.random((10, 10))
    MC_repetitions = 3
    result = compute_acquisition_fun(tunable_segmentation_method,
                                     img, img_sp, MC_repetitions)

    assert result is not None
    assert tunable_segmentation_method._run_pred.call_count == MC_repetitions


def test_compute_segmentation(tunable_segmentation_method):
    img = np.random.random((1, 1, 1, 10, 10, 3))
    labels_offset = 1
    result = compute_segmentation(tunable_segmentation_method, img,
                                  labels_offset)
    expected_segmentation = tunable_segmentation_method.segment(img)
    expected_segmentation = np.where(expected_segmentation,
                                     expected_segmentation + labels_offset,
                                     expected_segmentation)
    assert np.array_equal(result, expected_segmentation)
    assert tunable_segmentation_method._run_eval.called


def test_compute_acquisition(image_groups_manager, labels_manager,
                             tunable_segmentation_method,
                             make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.dims.axis_labels = ['t', 'z', 'y', 'x']

    acquisition_function = AcquisitionFunction(image_groups_manager,
                                               labels_manager,
                                               tunable_segmentation_method)

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


def test_add_multiscale_output_layer(single_scale_type_variant_array,
                                     simple_image_group,
                                     make_napari_viewer):
    image_group, _, _ = simple_image_group
    root_array, input_filename, data_group, _ = single_scale_type_variant_array
    output_filename = input_filename

    axes = "TCZYX"
    scale = [1, 1, 1, 1, 1]
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


def test_prepare_datasets_metadata(image_groups_manager, labels_manager,
                                   tunable_segmentation_method,
                                   simple_image_group,
                                   make_napari_viewer):
    image_group, _, _ = simple_image_group
    image_groups_manager.groups_root.addChild(image_group)

    viewer = make_napari_viewer()
    viewer.dims.axis_labels = ['t', 'z', 'y', 'x']

    acquisition_function = AcquisitionFunction(image_groups_manager,
                                               labels_manager,
                                               tunable_segmentation_method)

    acquisition_function._patch_sizes = {"T": 1, "X": 5, "Y": 5, "Z": 1}
    acquisition_function.input_axes = "TZYX"
    acquisition_function.model_axes = "YXC"

    # Define the input parameters for the method
    output_axes = "TCZYX"
    displayed_source_axes = "TCZYX"
    displayed_shape = [1, 3, 10, 10, 10]

    layers_group = image_group.child(0)
    layer_types = [(layers_group, "images")]

    # Call the method
    (dataset_metadata,
     sampling_positions) = acquisition_function._prepare_datasets_metadata(
         image_group,
         output_axes,
         displayed_source_axes,
         displayed_shape,
         layer_types
         )

    assert sampling_positions is None
    expected_dataset_metadata = {
        "images": {
            "filenames": layers_group.source_data,
            "data_group": layers_group.data_group,
            "source_axes": "TCZYX",
            "axes": "TZYXC",
            "roi": [(slice(None), slice(None), slice(0, 10), slice(0, 10),
                     slice(0, 10))],
            "modality": "images",
            'add_to_output': True
        }
    }
    for k, v in expected_dataset_metadata.items():
        assert dataset_metadata[k] == v


def test_compute_acquisition_layers(image_groups_manager, labels_manager,
                                    tunable_segmentation_method,
                                    make_napari_viewer,
                                    simple_image_group,
                                    labels_group,
                                    multiscale_layer_channel):

    image_group, _, _ = simple_image_group
    image_groups_manager.groups_root.addChild(image_group)

    viewer = make_napari_viewer()
    viewer.dims.axis_labels = ['t', 'z', 'y', 'x']

    with patch("napari_activelearning"
               "._acquisition"
               ".get_dataloader") as mock_dataloader, \
         patch("napari_activelearning"
               "._acquisition"
               ".add_multiscale_output_layer") as mock_add_ms_layer:
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

        mock_add_ms_layer.return_value = multiscale_layer_channel

        acquisition_function = AcquisitionFunction(
            image_groups_manager,
            labels_manager,
            tunable_segmentation_method)

        acquisition_function._patch_sizes = {"T": 1, "X": 5, "Y": 5, "Z": 1}
        acquisition_function.input_axes = "TZYX"
        acquisition_function.model_axes = "YXC"

        image_groups_manager.set_active_item(image_group)

        assert acquisition_function.compute_acquisition_layers(
            run_all=True,
            segmentation_group_name="segmentation",
            segmentation_only=False
        )

        image_group.setSelected(False)
        image_group.labels_group = None
        image_groups_manager.groups_root.takeChildren()
        image_groups_manager.groups_root.addChild(image_group)

        labels_manager.labels_group_root.takeChildren()
        labels_manager.labels_group_root.addChild(labels_group)


def test_fine_tune(image_groups_manager, simple_image_group,
                   labels_manager,
                   tunable_segmentation_method,
                   multiscale_layer_channel,
                   multiscale_layers_group,
                   labels_group,
                   make_napari_viewer):
    image_group, _, _ = simple_image_group
    image_groups_manager.groups_root.addChild(image_group)

    image_group.addChild(multiscale_layers_group)
    image_group.labels_group = labels_group

    viewer = make_napari_viewer()
    viewer.dims.axis_labels = ['t', 'z', 'y', 'x']

    with patch("napari_activelearning"
               "._acquisition"
               ".get_dataloader") as mock_dataloader, \
         patch("napari_activelearning"
               "._acquisition"
               ".add_multiscale_output_layer") as mock_add_ms_layer:
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

        mock_add_ms_layer.return_value = multiscale_layer_channel

        viewer.dims.axis_labels = ['t', 'z', 'y', 'x']
        acquisition_function = AcquisitionFunction(
            image_groups_manager,
            labels_manager,
            tunable_segmentation_method
        )

        acquisition_function._patch_sizes = {"T": 1, "X": 10, "Y": 10, "Z": 1}
        acquisition_function.input_axes = "TZYX"
        acquisition_function.model_axes = "YXC"

        assert acquisition_function.fine_tune()
        assert tunable_segmentation_method._fine_tune.called

    image_group.removeChild(multiscale_layers_group)
    image_group.labels_group = None
