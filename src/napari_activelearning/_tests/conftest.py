import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path, PureWindowsPath
import numpy as np
import zarr
import zarrdataset as zds

from napari.layers import Image
from napari.layers._source import Source
from napari.layers._multiscale_data import MultiScaleData

from napari_activelearning._layers import (LayerChannel,
                                           LayersGroup,
                                           ImageGroup,
                                           ImageGroupsManager)
from napari_activelearning._labels import LabelsManager, LabelGroup, LabelItem
from napari_activelearning._acquisition import AcquisitionFunction
from napari_activelearning._models import SimpleTunable


@pytest.fixture(scope="package")
def tunable_segmentation_method():
    method = SimpleTunable()
    method._run_pred = MagicMock(return_value=np.random.random((10, 10)))
    method._run_eval = MagicMock(return_value=np.random.randint(0, 2, (10, 10)))
    method._fine_tune = MagicMock(return_value=None)
    return method


@pytest.fixture(scope="package")
def output_temp_dir(tmpdir_factory):
    tmp_dir = tmpdir_factory.mktemp("temp")
    tmp_dir = Path(tmp_dir)
    yield tmp_dir


@pytest.fixture(scope="package", params=[Path, None, zarr.Group])
def output_group(request, tmpdir_factory):
    group_type = request.param
    if group_type is Path:
        tmp_dir = tmpdir_factory.mktemp("temp")
        zarr_group = Path(tmp_dir) / "output.zarr"
    elif group_type is zarr.Group:
        zarr_group = zarr.open()
    else:
        zarr_group = None

    yield zarr_group


@pytest.fixture(scope="package")
def single_scale_array():
    shape = (1, 3, 10, 10, 10)
    data = np.random.random(shape)
    return data, None, None, shape


@pytest.fixture(scope="package")
def multiscale_array(single_scale_array):
    data, _, _, shape = single_scale_array
    data = data[:, 0, ...]
    data = data * 255.0
    data = data.astype(np.int32)

    data = [data, data[..., ::2, ::2, ::2], data[..., ::4, ::4, ::4]]
    shape = [arr.shape for arr in data]

    return data, None, None, shape


@pytest.fixture(scope="package")
def single_scale_disk_zarr(single_scale_array, tmpdir_factory):
    tmp_dir = tmpdir_factory.mktemp("output")
    tmp_dir_path = Path(tmp_dir)

    sample_data, _, _, shape = single_scale_array

    input_filename = tmp_dir_path / "input.zarr"
    z_root = zarr.open(input_filename)

    data_group = "0"
    z_group = z_root.create_group(data_group)

    z_group.create_dataset(name="0", data=sample_data, overwrite=True)
    data_group = Path(data_group) / "0"

    if isinstance(data_group, PureWindowsPath):
        data_group = data_group.as_posix()

    data_group = str(data_group)

    return z_root, input_filename, data_group, shape


@pytest.fixture(scope="package")
def single_scale_memory_zarr(single_scale_array):

    sample_data, _, _, shape = single_scale_array

    z_root = zarr.open()

    data_group = "0"
    z_root.create_dataset(name=data_group, data=sample_data, overwrite=True)

    return z_root, None, data_group, shape


@pytest.fixture(scope="package")
def multiscale_disk_zarr(multiscale_array, tmpdir_factory):
    tmp_dir = tmpdir_factory.mktemp("output")
    tmp_dir_path = Path(tmp_dir)

    sample_data, _, _, shape = multiscale_array

    input_filename = tmp_dir_path / "input.zarr"
    z_root = zarr.open(input_filename)

    data_group = "0"
    z_group = z_root.create_group(data_group)

    source_data = []
    for lvl, data in enumerate(sample_data):
        z_group.create_dataset(name="%i" % lvl, data=data, overwrite=True)
        source_data.append(z_group["%i" % lvl])

    return source_data, input_filename, data_group, shape


@pytest.fixture(scope="package")
def dataset_metadata(single_scale_disk_zarr):
    z_root, input_filename, data_group, _ = single_scale_disk_zarr
    return {
        "images": {
            "filenames": [str(input_filename)],
            "data_group": data_group,
            "source_axes": "TCZYX",
            "axes": "TZYXC",
            "roi": None,
            "modality": "images"
        },
        "labels": {
            "filenames": [str(input_filename)],
            "data_group": data_group,
            "source_axes": "TCZYX",
            "axes": "TCZYX",
            "roi": None,
            "modality": "labels"
        }
    }


@pytest.fixture(scope="package")
def image_collection(single_scale_disk_zarr):
    source_data, input_filename, data_group, _ = single_scale_disk_zarr
    source_data = str(input_filename)

    collection = zds.ImageCollection(
        dict(
            images=dict(
                filename=source_data,
                data_group=data_group,
                source_axes="TCZYX",
                axes="TCZYX"
            )
        ),
        spatial_axes="ZYX"
    )
    return collection


@pytest.fixture(scope="function")
def single_scale_layer(single_scale_disk_zarr):
    source_data, input_filename, data_group, _ = single_scale_disk_zarr

    if isinstance(source_data, zarr.Group):
        source_data = source_data[data_group]

    layer = Image(
        data=source_data,
        name="sample_layer",
        scale=[1.0, 1.0, 1.0, 1.0],
        translate=[0.0, 0.0, 0.0, 0.0],
        visible=True
    )

    layer._source = Source(path=str(input_filename / data_group))

    return layer, source_data, input_filename, data_group


@pytest.fixture(scope="package")
def single_scale_memory_layer(single_scale_array):
    source_data, input_filename, data_group, _ = single_scale_array

    layer = Image(
        data=source_data,
        name="sample_memory_layer",
        scale=[1.0, 1.0, 1.0, 1.0],
        translate=[0.0, 0.0, 0.0, 0.0],
        visible=True
    )

    return layer, source_data, input_filename, data_group


@pytest.fixture(scope="function")
def multiscale_layer(multiscale_disk_zarr):
    source_data, input_filename, data_group, _ = multiscale_disk_zarr

    layer = Image(
        data=source_data,
        name="sample_segmentation_layer",
        scale=[1.0, 1.0, 1.0, 1.0],
        translate=[0.0, 0.0, 0.0, 0.0],
        visible=True
    )

    layer._source = Source(path=str(input_filename / data_group))

    data_group = str(Path(data_group) / "0")

    return layer, source_data, input_filename, data_group


@pytest.fixture(scope="function", params=[
    "single_scale_layer",
    "multiscale_layer"
])
def sample_layer(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="package", params=[
    "single_scale_array",
    "single_scale_memory_zarr",
    "single_scale_disk_zarr",
])
def single_scale_type_variant_array(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="function")
def layer_channel(single_scale_layer):
    layer, source_data, input_filename, data_group = single_scale_layer
    return LayerChannel(layer=layer, channel=1, source_axes="TCZYX")


@pytest.fixture(scope="function")
def multiscale_layer_channel(multiscale_layer):
    layer, source_data, input_filename, data_group = multiscale_layer
    return LayerChannel(layer=layer, channel=1, source_axes="TZYX")


@pytest.fixture(scope="function")
def multiscale_layers_group(multiscale_layer_channel):
    layers_group_mock = LayersGroup("segmentation",
                                    source_axes="TZYX",
                                    use_as_input_image=False,
                                    use_as_sampling_mask=False)
    layers_group_mock.addChild(multiscale_layer_channel)
    layers_group_mock.source_axes = "TZYX"

    return layers_group_mock


@pytest.fixture(scope="function")
def simple_image_group(single_scale_array):
    source_data, input_filename, data_group, _ = single_scale_array

    layer = Image(
        data=source_data,
        name="sample_layer",
        scale=[1.0, 1.0, 1.0, 1.0],
        translate=[0.0, 0.0, 0.0, 0.0],
        visible=True
    )

    image_group = ImageGroup("simple_group")

    layers_group = LayersGroup("simple_layers_group")
    image_group.addChild(layers_group)

    layer_channel = layers_group.add_layer(layer, 0, "TCZYX")
    image_group.input_layers_group = 0

    return image_group, layers_group, layer_channel


@pytest.fixture(scope="function")
def image_groups_manager():
    with patch('napari.current_viewer') as mock_viewer:
        mock_viewer.return_value.dims.axis_labels = ['t', 'z', 'y', 'x']
        mock_viewer.return_value.dims.ndim = 5
        mock_viewer.return_value.dims.ndisplay = 2

        image_groups_mgr = ImageGroupsManager()

    return image_groups_mgr


@pytest.fixture(scope="function")
def img_sampling_positions():
    sampling_positions = [
        LabelItem(
            0.01,
            position=(slice(0, 1), slice(3, 4), slice(0, 5), slice(0, 5))
        ),
        LabelItem(
            0.02,
            position=(slice(0, 1), slice(3, 4), slice(0, 5), slice(5, 10))
        ),
        LabelItem(
            0.03,
            position=(slice(0, 1), slice(3, 4), slice(5, 10), slice(0, 5))
        ),
        LabelItem(
            0.04,
            position=(slice(0, 1), slice(3, 4), slice(5, 10), slice(5, 10))
        )
    ]
    return sampling_positions


@pytest.fixture(scope="function")
def labels_group(layer_channel, img_sampling_positions):
    labels_group_mock = LabelGroup(layer_channel)
    labels_group_mock.addChildren(img_sampling_positions)
    return labels_group_mock


@pytest.fixture(scope="function")
def labels_manager(labels_group):
    with patch('napari.current_viewer') as mock_viewer:
        mock_viewer.return_value.dims.axis_labels = ['t', 'z', 'y', 'x']
        labels_manager_mock = LabelsManager()
        labels_manager_mock.labels_group_root.addChild(labels_group)

    return labels_manager_mock
