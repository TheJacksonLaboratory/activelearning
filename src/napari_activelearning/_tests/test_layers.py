import pytest
import operator

from napari.layers import Image
from napari_activelearning._layers import LayerChannel
import numpy as np


@pytest.fixture
def sample_layer():
    layer = Image(
        data=np.random.random((1, 10, 10, 10)),
        name="sample_layer",
        scale=[1.0, 1.0, 1.0, 1.0],
        translate=[0.0, 0.0, 0.0, 0.0],
        visible=True
    )

    return layer

@pytest.fixture
def layer_channel(sample_layer):
    return LayerChannel(layer=sample_layer, channel=1, source_axes="TZYX")


def test_initialization(layer_channel, sample_layer):
    assert layer_channel.layer == sample_layer
    assert layer_channel.channel == 1
    assert layer_channel.source_axes == "TZYX"
    assert layer_channel.name == "sample_layer"
    assert layer_channel.data_group is None
    assert np.array_equal(layer_channel.source_data, sample_layer.data)


def test_data_group(layer_channel):
    layer_channel.data_group = "test_group"
    assert layer_channel.data_group == "test_group"


def test_channel(layer_channel):
    layer_channel.channel = 2
    assert layer_channel.channel == 2


def test_source_axes(layer_channel):
    layer_channel.source_axes = "TCZYX"
    assert layer_channel.source_axes == "TZYX"


def test_name(layer_channel, sample_layer):
    layer_channel.name = "new_name"
    assert layer_channel.name == "new_name" and sample_layer.name == "new_name"
    sample_layer.name = "old_name"
    assert layer_channel.name == "old_name"


def test_shape(layer_channel, sample_layer):
    assert all(map(operator.eq, layer_channel.shape, sample_layer.data.shape))


def test_ndim(layer_channel, sample_layer):
    assert layer_channel.ndim == sample_layer.ndim


def test_scale(layer_channel):
    new_scale = [1.0, 2.0, 2.0, 2.0]
    layer_channel.scale = new_scale
    assert all(map(operator.eq, layer_channel.scale, new_scale))


def test_translate(layer_channel):
    new_translate = [0.0, 1.0, 1.0, 1.0]
    layer_channel.translate = new_translate
    assert all(map(operator.eq, layer_channel.translate, new_translate))


def test_visible(layer_channel):
    layer_channel.visible = False
    assert not layer_channel.visible


def test_selected(layer_channel, sample_layer, make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.layers.append(sample_layer)

    viewer.layers.selection.clear()

    layer_channel.selected = True
    assert sample_layer in viewer.layers.selection

    layer_channel.selected = False
    assert sample_layer not in viewer.layers.selection


def test_update_source_data(layer_channel, sample_layer):
    sample_layer.data = np.random.random((10, 10, 10))
    layer_channel._update_source_data()
    assert np.array_equal(layer_channel.source_data, sample_layer.data)
