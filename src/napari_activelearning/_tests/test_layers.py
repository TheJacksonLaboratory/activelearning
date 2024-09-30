import os
import shutil
from pathlib import Path

import operator

import numpy as np

from napari.layers import Labels
from napari.layers._source import Source

from napari_activelearning._layers import (LayerChannel,
                                           LayersGroup,
                                           ImageGroup,
                                           ImageGroupRoot,
                                           ImageGroupsManager,
                                           ImageGroupEditor,
                                           MaskGenerator)
from napari_activelearning._utils import get_source_data


def test_initialization(single_scale_layer):
    layer, _, input_filename, data_group = single_scale_layer

    layer_channel = LayerChannel(layer, 1, "TZYX")

    assert layer_channel.layer == layer
    assert layer_channel.channel == 1
    assert layer_channel.source_axes == "TZYX"
    assert layer_channel.name == layer.name
    assert layer_channel.data_group == data_group

    assert ((isinstance(layer_channel.source_data, (Path, str))
             and (Path(layer_channel.source_data.lower())
                  == Path(str(input_filename).lower())))
            or np.array_equal(layer_channel.source_data, layer.data))


def test_data_group(single_scale_layer):
    layer, _, input_filename, data_group = single_scale_layer

    layer_channel = LayerChannel(layer, 1, "TZYX")

    layer_channel.data_group = "test_group"
    assert layer_channel.data_group == "test_group"


def test_channel(single_scale_layer):
    layer, _, input_filename, data_group = single_scale_layer

    layer_channel = LayerChannel(layer, 1, "TZYX")

    layer_channel.channel = 2
    assert layer_channel.channel == 2


def test_source_axes(single_scale_layer):
    layer, _, input_filename, data_group = single_scale_layer

    layer_channel = LayerChannel(layer, 1, "TZYX")

    layer_channel.source_axes = "TCZYX"
    assert layer_channel.source_axes == "TCZYX"


def test_name(single_scale_layer):
    layer, _, _, _ = single_scale_layer

    layer_channel = LayerChannel(layer, 1, "TZYX")

    old_name = layer.name

    layer_channel.name = "new_name"
    assert layer_channel.name == "new_name" and layer.name == "new_name"

    layer.name = old_name
    assert layer_channel.name == old_name


def test_shape(single_scale_layer):
    layer, _, _, _ = single_scale_layer
    layer_channel = LayerChannel(layer, 1, "TZYX")

    assert all(map(operator.eq, layer_channel.shape, layer.data.shape))


def test_ndim(single_scale_layer):
    layer, _, _, _ = single_scale_layer
    layer_channel = LayerChannel(layer, 1, "TZYX")

    assert layer_channel.ndim == layer.ndim


def test_scale(single_scale_layer):
    layer, _, _, _ = single_scale_layer
    layer_channel = LayerChannel(layer, 1, "TZYX")

    old_scale = layer.scale

    new_scale = [1.0, 1.0, 2.0, 2.0, 2.0]
    layer_channel.scale = new_scale
    assert all(map(operator.eq, layer_channel.scale, new_scale))

    layer.scale = old_scale
    assert all(map(operator.eq, layer_channel.scale, old_scale))


def test_translate(single_scale_layer):
    layer, _, _, _ = single_scale_layer
    layer_channel = LayerChannel(layer, 1, "TZYX")

    old_translate = layer.translate

    new_translate = [0.0, 0.0, 1.0, 1.0, 1.0]
    layer_channel.translate = new_translate
    assert all(map(operator.eq, layer_channel.translate, new_translate))

    layer.translate = old_translate
    assert all(map(operator.eq, layer_channel.translate, old_translate))


def test_visible(single_scale_layer):
    layer, _, _, _ = single_scale_layer
    layer_channel = LayerChannel(layer, 1, "TZYX")

    layer.visible = False
    assert not layer_channel.visible

    layer_channel.visible = True
    assert layer.visible


def test_selected(single_scale_layer, make_napari_viewer):
    layer, _, _, _ = single_scale_layer
    layer_channel = LayerChannel(layer, 1, "TZYX")

    viewer = make_napari_viewer()
    viewer.layers.append(layer)

    viewer.layers.selection.clear()

    layer_channel.selected = True
    assert layer in viewer.layers.selection

    layer_channel.selected = False
    assert layer not in viewer.layers.selection


def test_update_source_data(single_scale_layer):
    layer, _, data_group, _ = single_scale_layer
    layer_channel = LayerChannel(layer, 1, "TZYX")

    old_data = layer.data
    old_source = layer._source.path

    layer.data = np.random.random((10, 10, 10))
    layer._source = Source(path=None)
    layer_channel._update_source_data()

    assert np.array_equal(layer_channel.source_data, layer.data)

    layer.data = old_data
    layer_channel.source_data = old_source
    assert layer._source.path == old_source


def test_layers_group_default_initialization():
    group = LayersGroup(layers_group_name="default_group")
    assert group.layers_group_name == "default_group"
    assert group.use_as_input_image is False
    assert group.use_as_sampling_mask is False
    assert group._source_axes_no_channels is None
    assert group._source_data is None
    assert group._data_group is None
    assert group.updated is True


def test_layers_group_properties(single_scale_layer, make_napari_viewer):
    layer, source_data, input_filename, data_group = single_scale_layer

    viewer = make_napari_viewer()
    viewer.layers.append(layer)

    layers_group = LayersGroup("sample_layers_group")
    layers_group.add_layer(layer, channel=0, source_axes="TCZYX")

    image_group = ImageGroup()
    image_group.addChild(layers_group)

    assert layers_group.layers_group_name == "sample_layers_group"
    layers_group.layers_group_name = "new_sample_layers_group"
    assert layers_group.layers_group_name == "new_sample_layers_group"
    assert image_group.group_name == "new_sample_layers_group"

    assert not layers_group.use_as_input_image
    assert not layers_group.use_as_sampling_mask

    layers_group.use_as_input_image = True
    assert layers_group.use_as_input_image
    layers_group.use_as_sampling_mask = True
    assert layers_group.use_as_sampling_mask

    layers_group.visible = True
    assert viewer.layers[layer.name].visible

    layers_group.visible = False
    assert not viewer.layers[layer.name].visible

    layers_group.selected = True
    assert layer in viewer.layers.selection

    layers_group.selected = False
    assert layer not in viewer.layers.selection

    assert all(map(operator.eq, layers_group.translate, layer.translate))
    assert all(map(operator.eq, layers_group.shape, layer.data.shape))
    assert all(map(operator.eq, layers_group.scale, layer.scale))

    assert layers_group.source_axes == "TCZYX"

    expected_metadata = {
        "modality": "new_sample_layers_group",
        "filenames": str(input_filename),
        "data_group": data_group,
        "source_axes": "TCZYX",
        "add_to_output": False
    }
    assert layers_group.metadata["modality"] == expected_metadata["modality"]
    assert (Path(layers_group.metadata["filenames"].lower())
            == Path(expected_metadata["filenames"].lower()))
    assert (layers_group.metadata["data_group"]
            == expected_metadata["data_group"])
    assert (layers_group.metadata["source_axes"]
            == expected_metadata["source_axes"])
    assert (layers_group.metadata["add_to_output"]
            == expected_metadata["add_to_output"])
    viewer.layers.clear()


def test_update_layers_group_source_data(single_scale_memory_layer,
                                         make_napari_viewer):
    layer, source_data, input_filename, data_group = single_scale_memory_layer

    viewer = make_napari_viewer()
    viewer.layers.append(layer)

    layers_group = LayersGroup("sample_layers_group")
    layers_group.add_layer(layer, 0, "TCZYX")
    layers_group.add_layer(layer, 1, "TCZYX")

    expected_array = np.concatenate((source_data, source_data), axis=1)

    assert all(map(operator.eq, layers_group.shape, expected_array.shape))
    assert all(map(operator.eq, layers_group.source_data.shape,
                   expected_array.shape))
    assert np.array_equal(layers_group.source_data, expected_array)


def test_update_layers_group_channels(single_scale_memory_layer,
                                      make_napari_viewer):
    layer, source_data, input_filename, data_group = single_scale_memory_layer

    viewer = make_napari_viewer()
    viewer.layers.append(layer)

    layers_group = LayersGroup("sample_layers_group")
    layer_channel_1 = layers_group.add_layer(layer, 0, "TCZYX")
    layer_channel_2 = layers_group.add_layer(layer, 1, "TCZYX")

    layers_group.move_channel(0, 1)

    assert layer_channel_1.channel == 1
    assert layer_channel_2.channel == 0

    layers_group.takeChild(1)
    assert layer_channel_1.channel == 0


def test_image_group_default_initialization():
    group = ImageGroup(group_name="default_image_group")
    assert group.group_name == "default_image_group"
    assert group.group_dir is None


def test_image_group_custom_initialization():
    group = ImageGroup(group_name="custom_image_group",
                       group_dir="/path/to/group")
    assert group.group_name == "custom_image_group"
    assert group.group_dir == Path("/path/to/group")


def test_children_image_group_root(make_napari_viewer):
    viewer = make_napari_viewer()

    group_root = ImageGroupRoot()
    image_group = ImageGroup("test_image_group")

    group_root.addChild(image_group)
    assert image_group.group_name in group_root.group_names

    group_root.removeChild(image_group)
    assert image_group.group_name not in group_root.group_names

    group_root.addChild(image_group)
    group_root.takeChild(0)
    assert image_group.group_name not in group_root.group_names

    group_root.addChild(image_group)
    group_root.takeChildren()
    assert not len(group_root.group_names)


def test_managed_layers_image_group_root(single_scale_memory_layer,
                                         make_napari_viewer):
    layer, source_data, input_filename, data_group = single_scale_memory_layer

    viewer = make_napari_viewer()
    viewer.layers.append(layer)

    group_root = ImageGroupRoot()

    image_group = ImageGroup("image_group")
    group_root.addChild(image_group)

    layers_group_1 = LayersGroup("layers_group_1")
    image_group.addChild(layers_group_1)

    layers_group_2 = LayersGroup("layers_group_2")
    image_group.addChild(layers_group_2)

    layer_channel_1 = layers_group_1.add_layer(layer, 0, "TCZYX")
    layer_channel_2 = layers_group_2.add_layer(layer, 0, "TCZYX")

    assert layer in group_root.managed_layers
    assert layer_channel_1 in group_root.managed_layers[layer]
    assert layer_channel_2 in group_root.managed_layers[layer]

    layers_group_2.removeChild(layer_channel_2)
    assert layer in group_root.managed_layers
    assert layer_channel_2 not in group_root.managed_layers[layer]

    viewer.layers.remove(layer)
    assert layer not in group_root.managed_layers
    assert not group_root.managed_layers


def test_image_group_manager_active_items(single_scale_memory_layer,
                                          make_napari_viewer):
    layer, source_data, input_filename, data_group = single_scale_memory_layer

    viewer = make_napari_viewer()
    viewer.dims.axis_labels = ['t', 'z', 'y', 'x']
    viewer.layers.append(layer)
    viewer.layers.selection.add(layer)

    manager = ImageGroupsManager("TZYX")
    manager.create_group()
    manager.update_group()

    image_group = manager.groups_root.child(0)
    layers_group = image_group.child(0)
    layer_channel = layers_group.child(0)

    manager.set_active_item(manager.groups_root)
    assert manager._active_layer_channel is None
    assert manager._active_layers_group is None
    assert manager._active_image_group is None
    assert manager.groups_root in manager.get_active_item()

    manager.set_active_item([])
    assert manager._active_layer_channel is None
    assert manager._active_layers_group is None
    assert manager._active_image_group is None
    assert manager.groups_root in manager.get_active_item()

    manager.set_active_item(image_group)
    assert manager._active_layer_channel is None
    assert manager._active_layers_group is None
    assert manager._active_image_group == image_group
    assert image_group in manager.get_active_item()

    manager.set_active_item(layers_group)
    assert manager._active_layer_channel is None
    assert manager._active_layers_group == layers_group
    assert manager._active_image_group == image_group
    assert layers_group in manager.get_active_item()

    manager.set_active_item(layer_channel)
    assert manager._active_layer_channel == layer_channel
    assert manager._active_layers_group == layers_group
    assert manager._active_image_group == image_group
    assert layer_channel in manager.get_active_item()

    manager.set_active_item([layers_group, layer_channel])
    assert manager._active_layer_channel == layer_channel
    assert manager._active_layers_group == layers_group
    assert manager._active_image_group == image_group
    assert (layer_channel in manager.get_active_item()
            and layers_group in manager.get_active_item())

    manager.set_active_item()
    assert manager._active_layer_channel == layer_channel
    assert manager._active_layers_group == layers_group
    assert manager._active_image_group == image_group
    assert layer_channel in manager.get_active_item()


def test_image_group_manager_focus(simple_image_group,
                                   make_napari_viewer):
    image_group, layers_group, layer_channel = simple_image_group
    layer = layer_channel.layer

    viewer = make_napari_viewer()
    viewer.layers.append(layer)
    viewer.layers.selection.add(layer)

    manager = ImageGroupsManager("TZYX")
    manager.groups_root.addChild(image_group)

    manager.focus_active_item([])
    assert len(viewer.layers.selection)

    manager.focus_active_item([None])
    assert len(viewer.layers.selection)

    manager.focus_active_item(image_group)
    assert layer_channel.layer.visible
    assert layers_group.selected
    assert layer_channel.selected
    assert image_group.selected

    manager.focus_active_item(manager.groups_root)
    assert not layer_channel.layer.visible


def test_image_group_manager_add_group(single_scale_memory_layer,
                                       make_napari_viewer):
    layer, source_data, input_filename, data_group = single_scale_memory_layer

    viewer = make_napari_viewer()
    viewer.dims.axis_labels = ['t', 'z', 'y', 'x']
    viewer.layers.append(layer)
    viewer.layers.selection.clear()

    manager = ImageGroupsManager("TZYX")
    manager.create_group()
    assert manager.groups_root.childCount() == 1
    image_group = manager.groups_root.child(0)
    assert image_group.childCount() == 0

    manager.set_active_item(manager.groups_root)
    manager.create_layers_group()
    assert image_group.childCount() == 0

    manager.set_active_item(image_group)
    manager.create_layers_group()
    assert image_group.childCount() == 1
    layers_group = image_group.child(0)

    viewer.layers.selection.clear()
    manager.set_active_item(layers_group)
    manager.add_layers_to_group()
    assert layers_group.childCount() == 0

    viewer.layers.selection.add(layer)
    manager.add_layers_to_group()
    assert layers_group.childCount() == 1

    manager.remove_layer()
    assert layers_group.childCount() == 0

    manager.remove_layers_group()
    assert image_group.childCount() == 0

    manager.remove_group()
    assert manager.groups_root.childCount() == 0

    viewer.layers.selection.clear()
    manager.create_group()
    image_group = manager.groups_root.child(0)
    assert image_group.childCount() == 0

    viewer.layers.selection.clear()
    manager.set_active_item(image_group)
    manager.update_group()
    assert image_group.childCount() == 0

    viewer.layers.selection.add(layer)
    manager.set_active_item(manager.groups_root)
    manager.update_group()
    assert image_group.childCount() == 0

    viewer.layers.selection.add(layer)
    manager.set_active_item(image_group)
    manager.update_group()
    assert image_group.childCount() == 1

    layers_group = image_group.child(0)
    assert layers_group.childCount() == 1


def test_image_group_manager_save_data(output_temp_dir,
                                       simple_image_group,
                                       make_napari_viewer):
    image_group, layers_group, layer_channel = simple_image_group

    layer = layer_channel.layer

    viewer = make_napari_viewer()
    viewer.dims.axis_labels = ['t', 'z', 'y', 'x']
    viewer.layers.append(layer)
    viewer.layers.selection.add(layer)

    manager = ImageGroupsManager("TZYX")
    manager.create_group()
    manager.add_layers_to_group()

    image_group = manager.groups_root.child(0)
    image_group.group_name = "test_group"
    image_group.group_dir = output_temp_dir

    layers_group = image_group.child(0)

    expected_image_file = output_temp_dir / "test_group.zarr"
    expected_metadata_file = output_temp_dir / "test_group_metadata.json"

    manager.set_active_item(manager.groups_root)
    manager.save_layers_group()
    assert not expected_image_file.exists()

    manager.set_active_item(layers_group)
    manager.save_layers_group()
    assert expected_image_file.exists()

    manager.dump_dataset_specs()
    assert expected_metadata_file.exists()

    shutil.rmtree(expected_image_file)
    os.remove(expected_metadata_file)


def test_update_reference_info(simple_image_group):
    image_group, layers_group, layer_channel = simple_image_group

    expected_im_shape = (10, 10, 10)
    expected_im_scale = (1, 1, 1)
    expected_im_translate = (0, 0, 0)
    expected_im_source_axes = "ZYX"

    mask_generator = MaskGenerator()

    mask_generator.active_image_group = image_group
    assert mask_generator.active_image_group == image_group
    assert mask_generator._update_reference_info() is True
    assert all(map(operator.eq, mask_generator._im_shape, expected_im_shape))
    assert all(map(operator.eq, mask_generator._im_scale, expected_im_scale))
    assert all(map(operator.eq, mask_generator._im_translate,
                   expected_im_translate))
    assert all(map(operator.eq, mask_generator._im_source_axes,
                   expected_im_source_axes))


def test_generate_mask_layer(simple_image_group, make_napari_viewer):
    image_group, layers_group, layer_channel = simple_image_group
    viewer = make_napari_viewer()

    mask_generator = MaskGenerator()
    mask_generator.active_image_group = image_group

    new_mask_layer = mask_generator.generate_mask_layer()

    assert isinstance(new_mask_layer, Labels)
    assert new_mask_layer in viewer.layers

    viewer.layers.clear()


def test_set_patch_size(simple_image_group):
    image_group, layers_group, layer_channel = simple_image_group
    mask_generator = MaskGenerator()

    mask_generator.set_patch_size([3, 16, 32])
    assert mask_generator._patch_sizes is None

    mask_generator.active_image_group = image_group

    mask_generator.set_patch_size([3, 16, 32])
    assert mask_generator._patch_sizes == [3, 16, 32]

    mask_generator.set_patch_size(64)
    assert mask_generator._mask_axes is not None
    assert mask_generator._patch_sizes == [64, 64, 64]


def test_image_group_editor_update_group_name():
    image_group_editor = ImageGroupEditor()

    image_group = ImageGroup()
    image_group_editor.active_image_group = image_group
    assert image_group.group_name is None

    image_group_editor.update_group_name()
    assert image_group.group_name is None

    image_group_editor.update_group_name("new_group_name")
    assert image_group.group_name == "new_group_name"


def test_image_group_editor_update_output_dir():
    image_group_editor = ImageGroupEditor()

    image_group = ImageGroup()
    image_group_editor.update_output_dir("/group/path/")
    assert image_group_editor._output_dir == "/group/path/"

    image_group_editor.active_image_group = image_group
    image_group_editor.update_output_dir("")
    assert image_group_editor._output_dir is None

    image_group_editor.update_output_dir("/group/path/")
    assert image_group.group_dir == Path("/group/path/")


def test_image_group_editor_update_channels(simple_image_group):
    image_group_editor = ImageGroupEditor()

    image_group, layers_group, layer_channel = simple_image_group

    image_group_editor.update_channels(1)
    assert image_group_editor._edit_channel is None

    image_group_editor.update_channels()
    assert image_group_editor._edit_channel is None

    image_group_editor.active_layer_channel = layer_channel
    image_group_editor.update_channels(1)
    assert layer_channel.channel == 1


def test_image_group_editor_update_use_as(simple_image_group,
                                          make_napari_viewer):
    viewer = make_napari_viewer()
    image_group_editor = ImageGroupEditor()

    image_group, layers_group, layer_channel = simple_image_group
    viewer.layers.append(layer_channel.layer)

    image_group_editor.update_use_as_input(False)
    assert image_group_editor._use_as_input is None

    image_group_editor.active_layers_group = layers_group
    image_group_editor.update_use_as_input(False)
    assert not image_group_editor._use_as_input
    assert image_group.input_layers_group is None

    image_group_editor.update_use_as_input(True)
    assert image_group_editor._use_as_input
    assert image_group.input_layers_group is not None


def test_image_group_editor_update_axes(simple_image_group,
                                        make_napari_viewer):
    viewer = make_napari_viewer()
    image_group_editor = ImageGroupEditor()

    image_group, layers_group, layer_channel = simple_image_group
    viewer.layers.append(layer_channel.layer)

    image_group_editor.update_source_axes("XYZ")
    assert image_group_editor._edit_axes is None

    image_group_editor.active_layer_channel = layer_channel
    image_group_editor.update_source_axes("XYZ")
    assert layer_channel.source_axes == "XYZ"

    assert all(map(operator.eq, ("0", "1", "x", "y", "z"),
                   viewer.dims.axis_labels))


def test_image_group_editor_update_translate(simple_image_group):
    image_group_editor = ImageGroupEditor()

    image_group, layers_group, layer_channel = simple_image_group

    image_group_editor = ImageGroupEditor()
    image_group_editor.update_translate([0, 0, 1, 1, 1])
    assert image_group_editor._edit_translate is None

    image_group_editor.active_layer_channel = layer_channel
    image_group_editor.update_translate([0, 0, 1, 1, 1])
    assert all(map(operator.eq, layer_channel.translate, [0, 0, 1, 1, 1]))


def test_image_group_editor_update_scale(simple_image_group):
    image_group_editor = ImageGroupEditor()

    image_group, layers_group, layer_channel = simple_image_group

    image_group_editor = ImageGroupEditor()
    image_group_editor.update_scale([1, 1, 2, 2, 2])
    assert image_group_editor._edit_scale is None

    image_group_editor.active_layer_channel = layer_channel
    image_group_editor.update_scale([1, 1, 2, 2, 2])
    assert all(map(operator.eq, layer_channel.scale, [1, 1, 2, 2, 2]))
