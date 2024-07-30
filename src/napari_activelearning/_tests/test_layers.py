from pathlib import Path
import operator
import numpy as np

from napari.layers._source import Source

from napari_activelearning._layers import LayerChannel


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
    layer, _, _, _ = single_scale_layer
    layer_channel = LayerChannel(layer, 1, "TZYX")

    old_data = layer.data
    old_source = layer._source

    layer.data = np.random.random((10, 10, 10))
    layer._source = Source(path=None)
    layer_channel._update_source_data()

    assert np.array_equal(layer_channel.source_data, layer.data)
    layer.data = old_data
    layer._source = old_source
