from pathlib import Path
import operator
import numpy as np

from napari.layers._source import Source

from napari_activelearning._layers import LayerChannel, LayersGroup, ImageGroup, ImageGroupRoot, ImageGroupsManager, ImageGroupEditor


def test_initialization(single_scale_layer):
    layer, _, input_filename, data_group = single_scale_layer

    layer_channel = LayerChannel(layer, 1, "TZYX")

    assert layer_channel.layer == layer
    assert layer_channel.channel == 1
    assert layer_channel.source_axes == "TZYX"
    assert layer_channel.name == layer.name
    assert layer_channel.data_group == data_group

    assert ((isinstance(layer_channel.source_data, (Path, str))
             and layer_channel.source_data == str(input_filename))
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


def test_selected(make_napari_viewer, single_scale_layer):
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


def test_layers_group_custom_initialization():
    group = LayersGroup(layers_group_name="custom_group",
                        source_axes="CXYZ",
                        use_as_input_image=True,
                        use_as_sampling_mask=True)
    assert group.layers_group_name == "custom_group"
    assert group.use_as_input_image is True
    assert group.use_as_sampling_mask is True
    assert group._source_axes == "XYZ"
    assert group._source_axes_no_channels == "XYZ"
    assert group._source_data is None
    assert group._data_group is None
    assert group.updated is True


def test_layers_group_properties(layers_group, single_scale_disk_zarr):
    z_root, input_filename, data_group, shape = single_scale_disk_zarr

    assert layers_group.layers_group_name == "sample_layers_group"
    assert layers_group.use_as_input_image
    assert not layers_group.use_as_sampling_mask
    assert layers_group.source_axes == "TCZYX"
    expected_metadata = {
            "modality": "sample_layers_group",
            "filenames": str(input_filename),
            "data_group": data_group,
            "source_axes": "TCZYX",
            "add_to_output": True
        }
    assert layers_group.metadata == expected_metadata


def test_update_layers_group_source_data(single_scale_memory_layer,
                                         make_napari_viewer):
    layer, source_data, input_filename, data_group = single_scale_memory_layer

    viewer = make_napari_viewer()
    viewer.layers.append(layer)

    layers_group = LayersGroup("sample_layers_group")
    layer_channel_1 = layers_group.add_layer(layer, 0, "TCZYX")
    layer_channel_2 = layers_group.add_layer(layer, 1, "TCZYX")

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


def test_managed_layers_image_group_root(layer_channel, make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.layers.append(layer_channel.layer)

    group_root = ImageGroupRoot()

    image_group = ImageGroup("image_group")
    group_root.addChild(image_group)

    layers_group_1 = LayersGroup("layers_group_1")
    image_group.addChild(layers_group_1)

    layers_group_2 = LayersGroup("layers_group_2")
    image_group.addChild(layers_group_2)

    layers_group_1.addChild(layer_channel)
    layers_group_2.addChild(layer_channel)

    assert layer_channel.layer in group_root.managed_layers
    assert layer_channel in group_root.managed_layers[layer_channel.layer]

    layers_group_2.removeChild(layer_channel)
    assert layer_channel.layer in group_root.managed_layers
    assert layer_channel in group_root.managed_layers[layer_channel.layer]

    viewer.layers.remove(layer_channel.layer)
    assert layer_channel.layer not in group_root.managed_layers
    assert not group_root.managed_layers


def test_image_group_manager_add_group(make_napari_viewer):
    viewer = make_napari_viewer()

    manager = ImageGroupsManager("TZYX")
    manager.create_group()
    assert manager.groups_root.childCount() == 1

    manager.create_layers_group()
    assert manager.groups_root.child(0).childCount() == 1

    manager.remove_layers_group()
    assert not manager.groups_root.child(0).childCount()

    manager.remove_group()
    assert not manager.groups_root.childCount()
