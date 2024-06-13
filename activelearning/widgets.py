from typing import List, Union

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
                            QLineEdit,
                            QLabel,
                            QFileDialog,
                            QSpinBox,
                            QProgressBar,
                            QTreeWidget,
                            QTreeWidgetItem,
                            QScrollArea,
                            QSplitter)

from functools import partial
from pathlib import Path
import torch
import napari
from napari.layers import Image, Labels
from napari.layers._multiscale_data import MultiScaleData
import tensorstore as ts
import numpy as np
import zarr

from ome_zarr.writer import write_multiscales_metadata, write_label_metadata

import zarrdataset as zds
import dask.array as da

import datautils


from cellpose import models, transforms


class DropoutEvalOverrider(torch.nn.Module):
    def __init__(self, dropout_module):
        super(DropoutEvalOverrider, self).__init__()

        self._dropout = type(dropout_module)(dropout_module.p,
                                             inplace=dropout_module.inplace)

    def forward(self, input):
        training_temp = self._dropout.training

        self._dropout.training = True
        out = self._dropout(input)

        self._dropout.training = training_temp

        return out


def add_dropout(net, p=0.05):
    # First step checks if there is any Dropout layer existing in the model
    has_dropout = False
    for module in net.modules():
        if isinstance(module, torch.nn.Sequential):
            for l_idx, layer in enumerate(module):
                if isinstance(layer, (torch.nn.Dropout, torch.nn.Dropout1d,
                                      torch.nn.Dropout2d,
                                      torch.nn.Dropout3d)):
                    has_dropout = True
                    break
            else:
                continue

            dropout_layer = module.pop(l_idx)
            module.insert(l_idx, DropoutEvalOverrider(dropout_layer))

    if has_dropout:
        return

    for module in net.modules():
        if isinstance(module, torch.nn.Sequential):
            for l_idx, layer in enumerate(module):
                if isinstance(layer, torch.nn.ReLU):
                    break
            else:
                continue

            dropout_layer = torch.nn.Dropout(p=p, inplace=True)
            module.insert(l_idx + 1, DropoutEvalOverrider(dropout_layer))


def compute_BALD(probs):
    if probs.ndim == 3:
        probs = np.stack((probs, 1 - probs), axis=1)

    T = probs.shape[0]

    probs_mean = probs.mean(axis=0)

    mutual_info = (-np.sum(probs_mean * np.log(probs_mean + 1e-12), axis=0)
                   + np.sum(probs * np.log(probs + 1e-12), axis=(0, 1)) / T)

    return mutual_info


def compute_acquisition(probs, super_pixel_labels):
    mutual_info = compute_BALD(probs)

    super_pixel_indices = np.unique(super_pixel_labels)

    u_sp_lab = np.zeros_like(super_pixel_labels, dtype=np.float32)

    for sp_l in super_pixel_indices:
        mask = super_pixel_labels == sp_l
        u_val = np.sum(mutual_info[mask]) / np.sum(mask)
        u_sp_lab = np.where(mask, u_val, u_sp_lab)

    return u_sp_lab


def cellpose_model_init(use_dropout=False):
    model = models.CellposeModel(gpu=False, model_type="cyto")
    model.mkldnn = False
    model.net.mkldnn = False

    if use_dropout:
        model.net.load_model(model.pretrained_model[0], device=model.device)
        model.net.eval()
        add_dropout(model.net)

    return model


def cellpose_probs(img, model):
    x = transforms.convert_image(img, None, normalize=False, invert=False,
                                 nchan=img.shape[-1])
    x = transforms.normalize_img(x, invert=False)

    x = torch.from_numpy(np.moveaxis(x, -1, 0))[None, ...]

    if x.shape[1] > 2:
        x = x[:, :2, ...]
    elif x.shape[1] < 2:
        x = torch.cat((x, x), dim=1)

    with torch.no_grad():
        y, _ = model.net(x)
        cellprob = y[0, 2, ...].detach().cpu()
        probs = cellprob.sigmoid().numpy()

    return probs


def cellpose_segment(img, model):
    seg, _, _ = model.eval(img, diameter=None, flow_threshold=None,
                           channels=[0, 0])
    return seg


def get_basename(layer_name):
    layer_base_name = layer_name.split(" ")
    if len(layer_base_name) > 1:
        layer_base_name = " ".join(layer_base_name[:-1])
    else:
        layer_base_name = layer_base_name[0]

    return layer_base_name


def compare_layers(layer, ref_layer=None, compare_type=True,
                   compare_shape=True):
    ref_layer_base_name = get_basename(ref_layer.name)

    if ref_layer_base_name not in layer.name:
        return False

    if (compare_type and isinstance(layer, type(ref_layer))
       and not layer._source.path == ref_layer._source.path):
        return False

    if compare_shape:
        ref_layer_axes = ref_layer.metadata.get("source_axes", None)
        layer_axes = layer.metadata.get("source_axes", None)

        if (ref_layer_axes
           and ref_layer.metadata.get("splitted_channels", False)):
            ref_layer_axes = [
                ax
                for ax in ref_layer_axes
                if ax != "C"
            ]

        if layer_axes and layer.metadata.get("splitted_channels", False):
            layer_axes = [
                ax
                for ax in layer_axes
                if ax != "C"
            ]

        if not (layer_axes and ref_layer_axes):
            if layer.data.shape != ref_layer.data.shape:
                return False

        else:
            layer_shape = {
                ax: round(ax_s * ax_scl)
                for ax, ax_s, ax_scl in zip(layer_axes, layer.data.shape,
                                            layer.scale)
            }

            ref_layer_shape = {
                ax: round(ax_s * ax_scl)
                for ax, ax_s, ax_scl in zip(ref_layer_axes,
                                            ref_layer.data.shape,
                                            ref_layer.scale)
            }

            if not all(layer_shape.get(ax, ax_s) == ax_s
                       for ax, ax_s in ref_layer_shape.items()):
                return False

    return True


def save_zarr(output_filename, data, shape, chunk_size, group_name, dtype):
    if output_filename is None:
        out_grp = zarr.group()
    elif isinstance(output_filename, (Path, str)):
        out_grp = zarr.open(output_filename, mode="a")
    elif isinstance(output_filename, zarr.Group):
        out_grp = output_filename
    else:
        raise ValueError(f"Output filename of type {type(output_filename)} is"
                         f" not supported")

    if not isinstance(chunk_size, bool) and isinstance(chunk_size, int):
        chunk_size = [chunk_size] * len(shape)

    if isinstance(chunk_size, list):
        chunks_size_axes = list(map(min, shape, chunk_size))
    else:
        chunks_size_axes = chunk_size

    out_grp.create_dataset(
        data=data,
        name=group_name,
        shape=shape,
        chunks=chunks_size_axes,
        compressor=zarr.Blosc(clevel=9),
        write_empty_chunks=False,
        dtype=dtype,
        overwrite=True
    )

    return out_grp


def downsample_image(z_root, source_axes, data_group, scale=4, num_scales=5,
                     reference_source_axes=None,
                     reference_scale=None,
                     reference_units=None):
    if isinstance(z_root, (Path, str)):
        source_arr = da.from_zarr(z_root, component=data_group)
    else:
        source_arr = da.from_zarr(z_root[data_group])

    z_ms = [source_arr]

    data_group = "/".join(data_group.split("/")[:-1])
    root_group = data_group + "/%i"

    source_arr_shape = {ax: source_arr.shape[source_axes.index(ax)]
                        for ax in source_axes}

    min_spatial_shape = min(source_arr_shape[ax]
                            for ax in "YX" if ax in source_axes)

    num_scales = min(num_scales, int(np.log(min_spatial_shape)
                                     / np.log(scale)))

    downscale_selection = tuple(
        slice(None, None, scale) if ax in "ZYX" and ax_s > 1 else slice(None)
        for ax, ax_s in zip(source_axes, source_arr.shape)
    )

    if reference_source_axes is None or reference_scale is None:
        reference_source_axes = source_axes
        reference_scale = [1.0] * len(source_axes)

    reference_scale_axes = {
        ax: ax_scl
        for ax, ax_scl in zip(reference_source_axes, reference_scale)
    }

    if not reference_units:
        reference_units = {
            ax: None
            for ax in reference_source_axes
        }

    datasets = [{
        "coordinateTransformations": [{
            "type": "scale",
            "scale": [reference_scale_axes.get(ax, 1.0) for ax in source_axes]
        }],
        "path": "0"
    }]

    for s in range(1, num_scales):
        target_arr = source_arr[downscale_selection]
        target_arr = target_arr.rechunk(
            tuple(
                tuple(np.ceil(chk / scale) for chk in chk_ax)
                for chk_ax in source_arr.chunks
            )
        )

        if isinstance(z_root, (Path, str)):
            target_arr.to_zarr(z_root,
                               component=root_group % s,
                               compressor=zarr.Blosc(clevel=9),
                               write_empty_chunks=False,
                               overwrite=True)

            source_arr = da.from_zarr(z_root, component=root_group % s)

            datasets.append({
                "coordinateTransformations": [
                    {"type": "scale",
                     "scale": [4.0 ** s * reference_scale_axes.get(ax, 1.0)
                               if ax in "ZYX" else 1.0
                               for ax in source_axes]
                     }],
                "path": str(s)
            })
        else:
            z_ms.append(target_arr)
            source_arr = target_arr

    if isinstance(z_root, Path):
        z_grp = zarr.open(z_root / data_group, mode="a")
        write_multiscales_metadata(z_grp, datasets,
                                   axes=list(source_axes.lower()))
        z_ms = z_root / data_group

    return z_ms


def get_source_data(layer):
    input_filename = layer._source.path
    data_group = ""

    if input_filename:
        input_filename = str(input_filename)
        data_group = "/".join(input_filename.split(".")[-1].split("/")[1:])
    else:
        return layer.data, None

    if data_group:
        input_filename = input_filename[:-len(data_group) - 1]

    if input_filename and isinstance(layer.data, MultiScaleData):
        data_group += "/0"

    if not input_filename:
        input_filename = None

    if not data_group:
        data_group = None

    return input_filename, data_group


class LayerChannel(QTreeWidgetItem):
    def __init__(self, layer, channel=0, source_axes="TZYX"):
        self.layer = layer
        self._channel = channel
        self._source_axes = source_axes

        super().__init__([layer.name, "", str(self._channel), source_axes])

        self._source_data = None
        self._data_group = None

    def _update_source_data(self):
        self._source_data, self._data_group = get_source_data(self.layer)
        if self._source_data is None:
            self._source_data = self.layer.data

    @property
    def source_data(self):
        if self._source_data is None:
            self._update_source_data()

        return self._source_data

    @property
    def data_group(self):
        if not self._data_group:
            self._update_source_data()

        return self._data_group

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, channel: int):
        self._channel = channel
        self.setText(2, str(self._channel))

    @property
    def source_axes(self):
        return self._source_axes

    @source_axes.setter
    def source_axes(self, source_axes: str):
        if "C" in source_axes:
            source_axes = list(source_axes)
            source_axes.remove("C")
            source_axes = "".join(source_axes)

        self._source_axes = source_axes
        self.setText(3, str(self._source_axes))


class LayerChannelGroup(QTreeWidgetItem):
    def __init__(self, layer_type: str,
                 layer_group_name: Union[str, None] = None,
                 source_axes: Union[str, None] = None):

        self._layer_group_name = layer_group_name
        self._layer_type = layer_type

        self._source_axes_no_channels = None
        self._source_axes = None
        self._source_data = None
        self._data_group = None

        self._updated = True
        self._metadata = None

        layer_group_name_str = layer_group_name
        source_axes_str = source_axes

        if not layer_group_name:
            layer_group_name_str = "Unset"

        if not source_axes:
            source_axes_str = "Unset"

        super().__init__([layer_group_name_str, layer_type, "",
                          source_axes_str])

        self.source_axes = source_axes

    def _update_source_axes(self):
        if not self._source_axes:
            return

        if "C" in self._source_axes:
            self._source_axes_no_channels = list(self._source_axes)
            self._source_axes_no_channels.remove("C")
            self._source_axes_no_channels = "".join(
                self._source_axes_no_channels
            )
        else:
            self._source_axes_no_channels = self._source_axes

        if "C" in self._source_axes and not self.childCount():
            self._source_axes = self._source_axes_no_channels

        elif "C" not in self._source_axes and self.childCount() > 1:
            self._source_axes = "C" + self._source_axes

        for idx in range(self.childCount()):
            self.child(idx).source_axes = self._source_axes_no_channels

    def _update_source_data(self):
        self._source_data = self.child(0).source_data
        self._data_group = self.child(0).data_group

        if not isinstance(self._source_data, (str, Path)):
            if "C" in self._source_axes:
                layers_channels = (
                    (self.child(idx).channel, self.child(idx))
                    for idx in range(self.childCount())
                )

                self._source_data = np.stack(
                    [layer_channel.source_data
                     for _, layer_channel in sorted(layers_channels)],
                    axis=self._source_axes.index("C")
                )

        self._updated = False

    @property
    def source_data(self):
        if self._source_data is None or self._updated:
            self._update_source_data()

        return self._source_data

    @property
    def data_group(self):
        if not self._data_group or self._updated:
            self._update_source_data()

        return self._data_group

    @property
    def metadata(self):
        metadata = {
            "modality": self._layer_type,
            "filenames": self.source_data,
            "data_group": self.data_group,
            "source_axes": self._source_axes,
            "add_to_output": self._layer_type in ("images", )
        }

        return metadata

    @property
    def layer_group_name(self):
        return self._layer_group_name

    @layer_group_name.setter
    def layer_group_name(self, layer_group_name: str):
        self._layer_group_name = layer_group_name
        self.setText(0, self._layer_group_name)

    @property
    def layer_type(self):
        return self._layer_type

    @layer_type.setter
    def layer_type(self, layer_type: str):
        self._layer_type = layer_type
        self.setText(1, self._layer_type)

    @property
    def source_axes(self):
        return self._source_axes

    @source_axes.setter
    def source_axes(self, source_axes: str):
        self._source_axes = source_axes
        self._update_source_axes()

        self.setText(3, self._source_axes)

    @property
    def shape(self):
        if not self.childCount():
            n_channels = len(self.source_axes) if self.source_axes else 0
            shape = [0] * n_channels

        else:
            shape = list(self.child(0).layer.data.shape)
            if self.childCount() > 1:
                shape.insert(0, self.childCount())

        return shape

    @property
    def scale(self):
        if not self.childCount():
            n_channels = len(self.source_axes) if self.source_axes else 0
            scale = [1] * n_channels

        else:
            scale = list(self.child(0).layer.scale)
            if self.childCount() > 1:
                scale.insert(0, 1)
            scale = tuple(scale)

        return scale

    def add_layer(self, layer, channel: Union[int, None] = None,
                  source_axes: Union[str, None] = None):
        if channel is None:
            channel = self.childCount()

        if source_axes is None:
            source_axes = self._source_axes_no_channels

        if not self._layer_group_name:
            self.layer_group_name = get_basename(layer.name)

        new_layer_channel = LayerChannel(layer, channel=channel,
                                         source_axes=source_axes)

        self.addChild(new_layer_channel)

        self.source_axes = source_axes

        self._updated = True

        return new_layer_channel

    def remove_layer(self, layer_channel: LayerChannel):
        removed_channel = layer_channel.channel

        for idx in range(self.childCount()):
            curr_layer_channel = self.child(idx)

            if curr_layer_channel.channel > removed_channel:
                curr_layer_channel.channel = curr_layer_channel.channel - 1

        self._update_source_axes()
        self._updated = True

    def takeChild(self, index: int):
        child = super(LayerChannelGroup, self).takeChild(index)
        self.remove_layer(child)
        return child

    def move_channel(self, from_channel: int, to_channel: int):
        channel_change = (-1) if from_channel < to_channel else 1

        left_channel = min(to_channel, from_channel)
        right_channel = max(to_channel, from_channel)

        for idx in range(self.childCount()):
            layer_channel = self.child(idx)

            if layer_channel.channel == from_channel:
                layer_channel.channel = to_channel

            elif left_channel <= layer_channel.channel <= right_channel:
                layer_channel.channel = layer_channel.channel + channel_change

        self.sortChildren(2, Qt.SortOrder.AscendingOrder)


class SamplingMaskGroup(LayerChannelGroup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_mask(self, name="sampling_mask", output_dir=Path(".")):
        output_filename = output_dir / (self.group_name + ".zarr")

        mask_data_group = "labels/" + name
        mask_source_data = self.source_data

        save_zarr(output_filename, data=mask_source_data,
                  shape=mask_source_data.shape,
                  chunk_size=True,
                  group_name=mask_data_group,
                  dtype=bool)

        mask_metadata = {
            "name": mask_data_group,
            "colors": [
                {
                    "label-value": 0,
                    "rgba": [0, 0, 0, 127]
                },
                {
                    "label-value": 1,
                    "rgba": [255, 255, 255, 127]
                }
            ],
            "properties": [
                {
                    "label-value": 0,
                    "class": "non sampleable"
                },
                {
                    "label-value": 1,
                    "class": "sampleable"
                },
            ]
        }

        z_grp = zarr.open(output_filename, mode="a")
        write_label_metadata(z_grp, **mask_metadata)

        self._source_data = str(output_filename)
        self._data_group = mask_data_group


class ImageGroup(QTreeWidgetItem):
    def __init__(self, group_name: Union[str, None] = None,
                 group_dir: Union[Path, str, None] = None):
        self._group_metadata = {}

        self._group_name = group_name
        self._group_dir = group_dir

        group_name_str = group_name
        group_dir_str = group_dir

        if not group_name:
            group_name_str = "Unset"

        if not group_dir:
            group_dir_str = "Unset"

        self._updated_groups = set()

        self._updated = False

        self._layer_types = []
        self._group_shapes = {}
        self._group_scales = {}
        self._group_source_axes = {}

        super().__init__([group_name_str, "", "", "", group_dir_str])

    @property
    def group_name(self):
        return self._group_name

    @group_name.setter
    def group_name(self, group_name: str):
        self._group_name = group_name
        self.setText(0, self._group_name)

    @property
    def group_dir(self):
        return self._group_dir

    @group_dir.setter
    def group_dir(self, group_dir: Union[Path, str]):
        if isinstance(group_dir, str):
            group_dir = Path(group_dir)

        self._group_dir = group_dir
        self.setText(4, str(self._group_dir))

    def _get_group_shapes(self):
        self._group_shapes = {}
        self._group_scales = {}
        self._group_source_axes = {}

        for idx in range(self.childCount()):
            layer = self.child(idx)

            self._group_shapes[layer.layer_type] = layer.shape
            self._group_scales[layer.layer_type] = layer.scale
            self._group_source_axes[layer.layer_type] = layer.source_axes

        self._updated = False

    @property
    def layer_type(self):
        if self._updated:
            self._get_group_shapes()

        return list(self._group_shapes.keys())

    @property
    def shape(self):
        if self._updated:
            self._get_group_shapes()

        return self._group_shapes

    @property
    def scale(self):
        if self._updated:
            self._get_group_shapes()

        return self._group_scales

    @property
    def source_axes(self):
        if self._updated:
            self._get_group_shapes()

        return self._group_source_axes

    def add_layer(self, layer_type, layer, channel: Union[int, None] = None,
                  source_axes: Union[str, None] = None):
        if not self._group_name:
            self.group_name = get_basename(layer.name)

        layer_group = list(filter(
            lambda curr_layer_group: curr_layer_group.layer_type == layer_type,
            map(lambda idx: self.child(idx), range(self.childCount()))
        ))

        if layer_group:
            layer_group = layer_group[0]

        if not layer_group:
            if "mask" in layer_type:
                layer_group = SamplingMaskGroup(layer_type,
                                                source_axes=source_axes)
            else:
                layer_group = LayerChannelGroup(layer_type,
                                                source_axes=source_axes)
            self.addChild(layer_group)

        layer_group.add_layer(layer, channel=channel,
                              source_axes=source_axes)

        self._updated_groups = self._updated_groups.union({layer_type})

        self._updated = True

        return layer_group

    def move_layer(self, layer_type: Union[str, None],
                   layer_group: LayerChannelGroup,
                   layer_channel: Union[LayerChannel, None] = None):
        if layer_channel:
            layer_channel_list = [
                layer_group.takeChild(layer_group.indexOfChild(layer_channel))
            ]

        else:
            layer_channel_list = layer_group.takeChildren()

        if layer_type:
            for curr_layer_channel in layer_channel_list:
                new_layer_group = self.add_layer(
                    layer_type,
                    curr_layer_channel.layer,
                    channel=None,
                    source_axes=curr_layer_channel.source_axes
                )
        else:
            new_layer_group = None

        if not layer_group.childCount():
            self.removeChild(layer_group)

        return new_layer_group

    def update_group_metadata(self):
        layer_groups = list(filter(
            lambda curr_layer_group:
            curr_layer_group.layer_type in self._updated_groups,
            map(lambda idx: self.child(idx), range(self.childCount()))
        ))

        for layer_group in layer_groups:
            self._group_metadata[layer_group.layer_type] = layer_group.metadata

        self._updated_groups.clear()

    @property
    def metadata(self):
        if self._group_metadata is None or self._updated_groups:
            self.update_group_metadata()

        return self._group_metadata


class ImageGroupEditor(QWidget):

    updated = Signal()

    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.viewer = viewer

        self.active_group = None
        self.active_layer_group = None
        self.active_layer = None

        self.group_name_lbl = QLabel("Group name:")
        self.group_name_le = QLineEdit("None selected")
        self.group_name_le.setEnabled(False)

        self.layer_name_lbl = QLabel("Layer name:")
        self.display_name_lbl = QLabel("None selected")

        self.layer_type_lbl = QLabel("Layer type:")
        self.layer_type_le = QLineEdit("None selected")
        self.layer_type_le.setEnabled(False)

        self.layer_axes_lbl = QLabel("Axes order:")
        self.edit_axes_le = QLineEdit("None selected")
        self.edit_axes_le.setEnabled(False)

        self.layer_channel_lbl = QLabel("Channel:")
        self.edit_channel_spn = QSpinBox(minimum=0, maximum=0)
        self.edit_channel_spn.setEnabled(False)

        self.remove_layer_group_btn = QPushButton("Remove layer group")
        self.remove_layer_group_btn.setEnabled(False)

        self.remove_layer_btn = QPushButton("Remove layer")
        self.remove_layer_btn.setEnabled(False)

        self.output_dir_lbl = QLabel("Output directory:")
        self.output_dir_le = QLineEdit("Unset")
        self.output_dir_dlg = QFileDialog(fileMode=QFileDialog.Directory)
        self.output_dir_btn = QPushButton("...")
        self.output_dir_le.setEnabled(False)
        self.output_dir_btn.setEnabled(False)

        self.remove_layer_group_btn.clicked.connect(self.remove_layer_group)
        self.remove_layer_btn.clicked.connect(self.remove_layer)

        self.output_dir_btn.clicked.connect(self.output_dir_dlg.show)
        self.output_dir_dlg.directoryEntered.connect(self._update_output_dir)

        self.layer_type_le.returnPressed.connect(self._update_metadata)
        self.group_name_le.returnPressed.connect(self._update_metadata)
        self.edit_channel_spn.editingFinished.connect(self._update_metadata)
        self.edit_channel_spn.valueChanged.connect(self._update_metadata)
        self.edit_axes_le.returnPressed.connect(self._update_metadata)
        self.output_dir_le.returnPressed.connect(self._update_metadata)

        self.edit_1_lyt = QHBoxLayout()
        self.edit_1_lyt.addWidget(self.group_name_lbl)
        self.edit_1_lyt.addWidget(self.group_name_le)

        self.edit_2_lyt = QHBoxLayout()
        self.edit_2_lyt.addWidget(self.layer_name_lbl)
        self.edit_2_lyt.addWidget(self.display_name_lbl)
        self.edit_2_lyt.addWidget(self.layer_type_lbl)
        self.edit_2_lyt.addWidget(self.layer_type_le)

        self.edit_3_lyt = QHBoxLayout()
        self.edit_3_lyt.addWidget(self.layer_axes_lbl)
        self.edit_3_lyt.addWidget(self.edit_axes_le)
        self.edit_3_lyt.addWidget(self.layer_channel_lbl)
        self.edit_3_lyt.addWidget(self.edit_channel_spn)

        self.edit_4_lyt = QHBoxLayout()
        self.edit_4_lyt.addWidget(self.output_dir_lbl)
        self.edit_4_lyt.addWidget(self.output_dir_le)
        self.edit_4_lyt.addWidget(self.output_dir_btn)

        self.edit_5_lyt = QHBoxLayout()
        self.edit_5_lyt.addWidget(self.remove_layer_btn)
        self.edit_5_lyt.addWidget(self.remove_layer_group_btn)

        self.edit_lyt = QVBoxLayout()
        self.edit_lyt.addLayout(self.edit_1_lyt)
        self.edit_lyt.addLayout(self.edit_2_lyt)
        self.edit_lyt.addLayout(self.edit_3_lyt)
        self.edit_lyt.addLayout(self.edit_4_lyt)
        self.edit_lyt.addLayout(self.edit_5_lyt)

        self.setLayout(self.edit_lyt)

    def _update_output_dir(self, output_dir):
        self.output_dir_le.setText(output_dir)

        self._update_metadata()

    def _update_metadata(self):
        new_group_name = self.group_name_le.text()
        new_layer_type = self.layer_type_le.text()
        new_source_axes = self.edit_axes_le.text()
        new_channel = self.edit_channel_spn.value()
        new_output_dir = self.output_dir_le.text()

        if new_output_dir.lower() in ("unset", "none", ""):
            new_output_dir = None

        display_source_axes = list(new_source_axes.lower())
        if "c" in display_source_axes:
            display_source_axes.remove("c")
        display_source_axes = tuple(display_source_axes)

        if display_source_axes != self.viewer.dims.axis_labels:
            self.viewer.dims.axis_labels = display_source_axes

        if self.active_layer:
            if self.active_layer.source_axes != new_source_axes:
                self.active_layer.source_axes = new_source_axes

            prev_channel = self.active_layer.channel
            if prev_channel != new_channel:
                self.active_layer_group.move_channel(prev_channel, new_channel)

        if self.active_layer_group:
            if self.active_layer_group.source_axes != new_source_axes:
                self.active_layer_group.source_axes = new_source_axes

            if self.active_layer_group.layer_type != new_layer_type:
                self.active_layer_group = self.active_group.move_layer(
                    new_layer_type,
                    self.active_layer_group,
                    layer_channel=self.active_layer
                )

                self.active_layer = None

        if self.active_group:
            if self.active_group.group_name != new_group_name:
                self.active_group.group_name = new_group_name

            if self.active_group.group_dir != new_output_dir:
                self.active_group.group_dir = new_output_dir

        self._update_edit_box()
        self.updated.emit()

    def _update_edit_box(self):
        if self.active_group:
            self.output_dir_le.setText(str(self.active_group.group_dir))
            self.group_name_le.setText(self.active_group.group_name)

            self.output_dir_btn.setEnabled(True)
            self.output_dir_le.setEnabled(True)
            self.group_name_le.setEnabled(True)
        else:
            self.output_dir_le.setText("None selected")
            self.group_name_le.setText("None selected")

            self.output_dir_btn.setEnabled(False)
            self.output_dir_le.setEnabled(False)
            self.group_name_le.setEnabled(False)

        if self.active_layer_group:
            self.layer_type_le.setText(self.active_layer_group.layer_type)
            self.edit_axes_le.setText(self.active_layer_group.source_axes)
            self.remove_layer_group_btn.setEnabled(True)
            self.layer_type_le.setEnabled(True)
            self.edit_axes_le.setEnabled(True)
        else:
            self.layer_type_le.setText("None selected")
            self.edit_axes_le.setText("None selected")
            self.remove_layer_group_btn.setEnabled(False)
            self.layer_type_le.setEnabled(False)
            self.edit_axes_le.setEnabled(False)

        if self.active_layer:
            self.display_name_lbl.setText(self.active_layer.layer.name)
            self.edit_axes_le.setText(self.active_layer.source_axes)
            self.remove_layer_btn.setEnabled(True)
            self.edit_axes_le.setEnabled(True)

            self.edit_channel_spn.setMaximum(
                self.active_layer_group.childCount() - 1
            )
            self.edit_channel_spn.setValue(self.active_layer.channel)
            self.edit_channel_spn.setEnabled(True)

        else:
            self.display_name_lbl.setText("None selected")

            self.edit_channel_spn.setValue(0)
            self.edit_channel_spn.setMaximum(0)
            self.edit_channel_spn.setEnabled(False)
            self.remove_layer_btn.setEnabled(False)

    def remove_layer_group(self):
        if not self.active_layer_group:
            return

        self.active_group.move_layer(None, self.active_layer_group,
                                     layer_channel=None)
        self._update_edit_box()

    def remove_layer(self):
        if not (self.active_layer_group and self.active_layer):
            return

        self.active_group.move_layer(None, self.active_layer_group,
                                     layer_channel=self.active_layer)

        self.active_layer = None
        self._update_edit_box()

    def setActiveLayerGroup(self, active_group: Union[ImageGroup, None] = None,
                            active_layer_group: Union[LayerChannelGroup, None] = None,
                            active_layer: Union[LayerChannel, None] = None):
        self.active_group = active_group
        self.active_layer_group = active_layer_group
        self.active_layer = active_layer
        self._update_edit_box()


class MaskGenerator(QWidget):

    mask_created = Signal()

    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.viewer = viewer
        self.active_layer_group = None

        self.patch_size_lbl = QLabel("Patch size")
        self.patch_size_spn = QSpinBox(minimum=128, maximum=4096,
                                       singleStep=16)
        self.patch_size_spn.setEnabled(False)

        self.generate_mask_btn = QPushButton("Create mask")
        self.generate_mask_btn.setToolTip("Create a napari Label layer with a "
                                          "blank mask at the scale selected "
                                          "with `Patch size`")
        self.generate_mask_btn.setEnabled(False)
        self.generate_mask_btn.clicked.connect(self.generate_mask_layer)

        self.create_mask_lyt = QHBoxLayout()
        self.create_mask_lyt.addWidget(self.patch_size_lbl)
        self.create_mask_lyt.addWidget(self.patch_size_spn)
        self.create_mask_lyt.addWidget(self.generate_mask_btn)

        self.setLayout(self.create_mask_lyt)

    def setActiveLayerGroup(self, active_layer_group):
        self.active_layer_group = active_layer_group

        self.generate_mask_btn.setEnabled(self.active_layer_group is not None)
        self.patch_size_spn.setEnabled(self.active_layer_group is not None)

    def generate_mask_layer(self):
        if (not self.active_layer_group
           or not self.active_layer_group.source_axes):
            return

        patch_size = self.patch_size_spn.value()

        source_axes = [
            ax
            for ax in self.active_layer_group.source_axes
            if ax != "C"
        ]

        reference_layer = self.active_layer_group.child(0).layer
        im_shape = reference_layer.data.shape
        im_scale = reference_layer.scale
        im_trans = reference_layer.translate

        mask_axes = "".join([
            ax
            for ax in source_axes
            if ax in "ZYX"
        ])

        mask_shape = [int(np.ceil(s / patch_size))
                      for s, ax in zip(im_shape, source_axes)
                      if ax in mask_axes]

        mask_scale = tuple(
            (ax_scl * patch_size) if ax_s // patch_size > 1 else ax_scl
            for ax, ax_s, ax_scl in zip(source_axes, im_shape, im_scale)
            if ax in mask_axes
        )

        mask_translate = tuple(
            ax_trans
            + ((ax_scl * (patch_size - 1) / 2)
               if ax_s // patch_size > 1 else 0)
            for ax, ax_s, ax_scl, ax_trans in zip(source_axes, im_shape,
                                                  im_scale,
                                                  im_trans)
            if ax in mask_axes
        )

        new_mask_layer = self.viewer.add_labels(
            data=np.zeros(mask_shape, dtype=np.uint8),
            scale=mask_scale,
            translate=mask_translate,
            name=self.active_layer_group.layer_group_name + " sample-mask",
        )

        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(new_mask_layer)

        self.mask_created.emit()


class ImageGroupsManager(QWidget):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer

        self.active_group = None
        self.active_layer_group = None
        self.active_layer = None

        self.image_groups_tw = QTreeWidget()

        self.image_groups_tw.setColumnCount(4)
        self.image_groups_tw.setHeaderLabels(["Group name", "Layer type",
                                              "Channels",
                                              "Axes order"])

        self.image_groups_tw.itemSelectionChanged.connect(
            self._get_active_item
        )
        self.root_group = QTreeWidgetItem(["All image groups"])
        self.image_groups_tw.addTopLevelItem(self.root_group)

        self.new_group_btn = QPushButton("New")
        self.new_group_btn.setToolTip("Create a new group. If layers are "
                                      "selected, add these to the new group.")
        self.new_group_btn.clicked.connect(self.create_group)

        self.add_group_btn = QPushButton("Add")
        self.add_group_btn.setEnabled(False)
        self.add_group_btn.setToolTip("Add selected layers to current group")
        self.add_group_btn.clicked.connect(self.update_group)

        self.remove_group_btn = QPushButton("Remove")
        self.remove_group_btn.setEnabled(False)
        self.remove_group_btn.setToolTip("Remove selected group. This will not"
                                         " remove the layers from napari "
                                         "viewer.")
        self.remove_group_btn.clicked.connect(self.remove_group)

        self.buttons_lyt = QHBoxLayout()
        self.buttons_lyt.addWidget(self.new_group_btn)
        self.buttons_lyt.addWidget(self.add_group_btn)
        self.buttons_lyt.addWidget(self.remove_group_btn)

        self.layers_editor = ImageGroupEditor(viewer)

        self.mask_creator = MaskGenerator(viewer)
        self.mask_creator.mask_created.connect(self.update_group)

        self.group_lyt = QVBoxLayout()
        self.group_lyt.addLayout(self.buttons_lyt)
        self.group_lyt.addWidget(self.layers_editor)
        self.group_lyt.addWidget(self.mask_creator)
        self.group_lyt.addWidget(self.image_groups_tw)

        self.setLayout(self.group_lyt)

    def _get_active_item(self):
        item = self.image_groups_tw.selectedItems()
        if isinstance(item, list):
            item = item[0]

        self.active_layer = None
        self.active_layer_group = None
        self.active_group = None

        if isinstance(item, LayerChannel):
            self.active_layer = item
            self.active_layer_group = item.parent()
            self.active_group = item.parent().parent()

        elif isinstance(item, LayerChannelGroup):
            self.active_layer_group = item
            self.active_group = item.parent()

        elif isinstance(item, ImageGroup) and item != self.root_group:
            self.active_group = item

        self.mask_creator.setActiveLayerGroup(self.active_layer_group)
        self.layers_editor.setActiveLayerGroup(
            active_group=self.active_group,
            active_layer_group=self.active_layer_group,
            active_layer=self.active_layer
        )

        self.remove_group_btn.setEnabled(self.active_group is not None)
        self.add_group_btn.setEnabled(self.active_group is not None)

    def update_group(self):
        self._updated = True

        selected_layers = list(sorted(
            map(lambda layer: (layer.name, layer),
                self.viewer.layers.selection)
        ))
        if not selected_layers or not self.active_group:
            return

        active_source_axes = "".join(self.viewer.dims.axis_labels).upper()
        if self.active_layer_group:
            active_source_axes = self.active_layer_group.source_axes

        if active_source_axes:
            if "C" in active_source_axes:
                active_source_axes = list(active_source_axes)
                active_source_axes.remove("C")
                active_source_axes = "".join(active_source_axes)

        group_names = set()

        for layer_name, layer in selected_layers:
            if not self.active_group.group_name:
                if layer_name in group_names:
                    n_existing = sum(map(
                        lambda exisitng_group_name:
                        layer_name in exisitng_group_name,
                        group_names
                    ))

                    layer_name = layer_name + " (%i)" % n_existing

                group_names = group_names.union({layer_name})
                self.active_group.group_name = layer_name

            layer_type = "images" if isinstance(layer, Image) else "masks"
            self.active_group.add_layer(
                layer_type, layer, channel=None, source_axes=active_source_axes
            )

    def create_group(self):
        self.active_group = ImageGroup()
        self.root_group.addChild(self.active_group)

        self.update_group()

    def remove_group(self):
        self.root_group.removeChild(self.active_group)

        self.active_layer = None
        self.active_layer_group = None
        self.active_group = None

        self._updated = True


class LabelsManager(QWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()

        self.viewer = viewer

        self._current_patch_id = 0
        self._current_image_id = 0

        self.prev_img_btn = QPushButton('<<')
        self.prev_img_btn.clicked.connect(self._previus_image)

        self.prev_patch_btn = QPushButton('<')
        self.prev_patch_btn.clicked.connect(self._previus_patch)

        self.next_patch_btn = QPushButton('>')
        self.next_patch_btn.clicked.connect(self._next_patch)

        self.next_img_btn = QPushButton('>>')
        self.next_img_btn.clicked.connect(self._next_image)

        self.fix_labels_btn = QPushButton("Fix current labels")
        self.fix_labels_btn.clicked.connect(self.fix_labels)

        self.nav_btn_layout = QHBoxLayout()
        self.nav_btn_layout.addWidget(self.prev_img_btn)
        self.nav_btn_layout.addWidget(self.prev_patch_btn)
        self.nav_btn_layout.addWidget(self.next_patch_btn)
        self.nav_btn_layout.addWidget(self.next_img_btn)

        self.fix_btn_layout = QVBoxLayout()
        self.fix_btn_layout.addWidget(self.fix_labels_btn)
        self.fix_btn_layout.addLayout(self.nav_btn_layout)

        self.setLayout(self.fix_btn_layout)

        self.txn = None
        self.layer_name = None

        self.sampling_positions = {}

        self.viewer.layers.events.removing.connect(
            self._remove_sampling_positions
        )

    def _remove_sampling_positions(self, event):
        for removed_layers in event._sources[0]:
            output_name = removed_layers.name

            # Remove any invalid character from the name
            for chr in [" ", ".", "/", "\\"]:
                output_name = "-".join(output_name.split(chr))

            if output_name in self.sampling_positions:
                index = self.sampling_positions.index(output_name)
                self.sampling_positions.pop(output_name)

                if self._current_image_id > index:
                    self._current_image_id -= 1
                elif self._current_image_id == index:
                    self._next_image()

    def _previus_image(self):
        self.navigate(delta_image_index=-1)

    def _previus_patch(self):
        self.navigate(delta_patch_index=-1)

    def _next_image(self):
        self.navigate(delta_image_index=1)

    def _next_patch(self):
        self.navigate(delta_patch_index=1)

    def navigate(self, delta_image_index=0, delta_patch_index=0):
        if self.txn:
            self.txn.commit_async()
            self.viewer.layers.remove(self.viewer.layers["Labels edit"])
            self.viewer.layers[self.layer_name].refresh()

        if not len(self.sampling_positions):
            return

        layer_names = list(self.sampling_positions.keys())

        self._current_image_id += delta_image_index
        self._current_patch_id += delta_patch_index
        self.layer_name = layer_names[self._current_image_id]

        if delta_patch_index:
            if self._current_patch_id < 0:
                self._current_image_id -= 1

            elif (len(self.sampling_positions[self.layer_name])
                  <= self._current_patch_id):
                self._current_image_id += 1
                self._current_patch_id = 0

        if self._current_image_id < 0:
            self._current_image_id = len(layer_names) - 1

        elif self._current_image_id >= len(layer_names):
            self._current_image_id = 0

        self.layer_name = layer_names[self._current_image_id]

        current_position_list = \
            self.sampling_positions[self.layer_name]

        if self._current_patch_id < 0:
            self._current_patch_id = len(current_position_list) - 1

        self.current_position = \
            current_position_list[self._current_patch_id]

        segmentation_layer = self.viewer.layers[self.layer_name]
        current_center = [
            (ax_roi.stop + ax_roi.start) / 2 * ax_scl
            for ax_roi, ax_scl in zip(self.current_position,
                                      segmentation_layer.scale)
        ]

        self.viewer.dims.order = tuple(range(self.viewer.dims.ndim))
        self.viewer.camera.center = (
            *self.viewer.camera.center[:-len(current_center)],
            *current_center
        )
        self.viewer.camera.zoom = 1 / min(segmentation_layer.scale)

        # Toggle visibility to show the current segmentation layer and all
        # layers related to it.
        reference_layer_name = self.layer_name.split(" segmentation output")[0]
        reference_layer = self.viewer.layers[reference_layer_name]

        sibling_layers = list(filter(
            partial(compare_layers, ref_layer=reference_layer,
                    compare_type=False,
                    compare_shape=True),
            self.viewer.layers
        ))

        for curr_layer in self.viewer.layers:
            curr_layer.visible = curr_layer in sibling_layers

    def fix_labels(self):
        if not self.layer_name:
            return

        segmentation_layer = self.viewer.layers[self.layer_name]

        input_filename = segmentation_layer._source.path
        data_group = ""

        if input_filename:
            input_filename = str(input_filename)
            data_group = "/".join(input_filename.split(".")[-1].split("/")[1:])

        if data_group:
            input_filename = input_filename[:-len(data_group) - 1]

        if isinstance(segmentation_layer.data, MultiScaleData):
            data_group += "/0"

        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': str(Path(input_filename) / data_group),
            },
        }

        ts_array = ts.open(spec).result()

        self.txn = ts.Transaction()

        lazy_data = ts_array.with_transaction(self.txn)
        lazy_data = lazy_data[ts.d[:][self.current_position].translate_to[0]]

        self.viewer.add_labels(lazy_data, name="Labels edit",
                               blending="translucent_no_depth",
                               opacity=0.7,
                               translate=[ax_roi.start
                                          for ax_roi in self.current_position])
        self.viewer.layers["Labels edit"].bounding_box.visible = True


class AcquisitionFunction(QWidget):
    def __init__(self, viewer: napari.Viewer, labels_manager: LabelsManager,
                 image_groups_manager: ImageGroupsManager):
        super().__init__()

        self.viewer = viewer

        self.patch_size_lbl = QLabel("Patch size:")
        self.patch_size_spn = QSpinBox(minimum=128, maximum=1024,
                                       singleStep=128)
        self.patch_size_lyt = QHBoxLayout()
        self.patch_size_lyt.addWidget(self.patch_size_lbl)
        self.patch_size_lyt.addWidget(self.patch_size_spn)

        self.max_samples_lbl = QLabel("Maximum samples:")
        self.max_samples_spn = QSpinBox(minimum=1, maximum=10000, value=100,
                                        singleStep=10)
        self.max_samples_lyt = QHBoxLayout()
        self.max_samples_lyt.addWidget(self.max_samples_lbl)
        self.max_samples_lyt.addWidget(self.max_samples_spn)

        self.MC_repetitions_lbl = QLabel("Monte Carlo repetitions")
        self.MC_repetitions_spn = QSpinBox(minimum=1, maximum=100, value=30,
                                           singleStep=10)
        self.MC_repetitions_lyt = QHBoxLayout()
        self.MC_repetitions_lyt.addWidget(self.MC_repetitions_lbl)
        self.MC_repetitions_lyt.addWidget(self.MC_repetitions_spn)

        self.execute_btn = QPushButton("Run on selected layers")
        self.execute_btn.clicked.connect(self.compute_acquisition_layers)

        self.image_lbl = QLabel("Image queue:")
        self.image_pb = QProgressBar()
        self.image_lyt = QHBoxLayout()
        self.image_lyt.addWidget(self.image_lbl)
        self.image_lyt.addWidget(self.image_pb)

        self.patch_lbl = QLabel("Patch queue:")
        self.patch_pb = QProgressBar()
        self.patch_lyt = QHBoxLayout()
        self.patch_lyt.addWidget(self.patch_lbl)
        self.patch_lyt.addWidget(self.patch_pb)

        self.acquisition_lyt = QVBoxLayout()
        self.acquisition_lyt.addLayout(self.patch_size_lyt)
        self.acquisition_lyt.addLayout(self.max_samples_lyt)
        self.acquisition_lyt.addLayout(self.MC_repetitions_lyt)
        self.acquisition_lyt.addWidget(self.execute_btn)
        self.acquisition_lyt.addLayout(self.image_lyt)
        self.acquisition_lyt.addLayout(self.patch_lyt)
        self.setLayout(self.acquisition_lyt)

        self.labels_manager = labels_manager
        self.image_groups_manager = image_groups_manager

    def _update_roi_from_position(self):
        displayed_axes = "".join(self.viewer.dims.axis_labels).upper()
        position = self.viewer.cursor.position
        axes_order = self.viewer.dims.order

        roi_start = [0] * len(axes_order)
        roi_length = [-1] * len(axes_order)
        for ord in axes_order[:-self.viewer.dims.ndisplay]:
            roi_start[ord] = int(position[ord])
            roi_length[ord] = 1

        self._roi = {
            ax: slice(ax_start if ax_length > 0 else None,
                      (ax_start + ax_length) if ax_length > 0 else None)
            for ax, ax_start, ax_length in zip(displayed_axes, roi_start,
                                               roi_length)
        }

    @staticmethod
    def compute_acquisition_fun(dataset_metadata,
                                acquisition_fun,
                                segmentation_out,
                                MC_repetitions=30,
                                max_samples=1000,
                                patch_size=128):
        dl = datautils.get_dataloader(dataset_metadata, patch_size=patch_size,
                                      shuffle=True)

        model_dropout = cellpose_model_init(use_dropout=True)
        model = cellpose_model_init(use_dropout=False)

        acquisition_fun_max = 0
        segmentation_max = 0
        n_samples = 0
        img_sampling_positions = []

        for pos, img, img_sp in dl:
            probs = []
            for _ in range(MC_repetitions):
                probs.append(
                    cellpose_probs(img[0].numpy(), model_dropout)
                )
            probs = np.stack(probs, axis=0)

            u_sp_lab = compute_acquisition(probs, img_sp[0, ..., 0].numpy())

            pos_u_lab = (slice(pos[0, 0, 0].item(), pos[0, 0, 1].item()),
                         slice(pos[0, 1, 0].item(), pos[0, 1, 1].item()))

            acquisition_fun[pos_u_lab] = u_sp_lab
            acquisition_fun_max = max(acquisition_fun_max, u_sp_lab.max())

            # Execute segmentation once to get the expected labels
            seg_out = cellpose_segment(img[0].numpy(), model)
            seg_out = np.where(seg_out, seg_out + segmentation_max, 0)
            segmentation_out[pos_u_lab] = seg_out

            segmentation_max = max(segmentation_max, seg_out.max())

            img_sampling_positions.append(pos_u_lab)

            n_samples += 1

            if n_samples >= max_samples:
                break

        return acquisition_fun_max

    def compute_acquisition_layers(self):
        MC_repetitions = self.MC_repetitions_spn.value()
        max_samples = self.max_samples_spn.value()
        patch_size = self.patch_size_spn.value()

        self._update_roi_from_position()

        image_groups = list(map(
            lambda idx: self.image_groups_manager.root_group.child(idx),
            range(self.image_groups_manager.root_group.childCount())
        ))

        self.image_pb.setRange(0, len(image_groups))
        self.image_pb.reset()

        for n, image_group in enumerate(image_groups):
            group_name = image_group.group_name
            if image_group.group_dir:
                output_filename = image_group.group_dir / (group_name
                                                           + ".zarr")
            else:
                output_filename = None

            displayed_source_axes = image_group.source_axes["images"]
            displayed_shape = image_group.shape["images"]
            displayed_scale = image_group.scale["images"]

            acquisition_fun_shape = [
                ax_s
                for ax, ax_s in zip(displayed_source_axes,
                                    displayed_shape)
                if ax in "ZYX" and ax_s > 1
            ]

            acquisition_fun_scale = [
                ax_scl
                for ax, ax_s, ax_scl in zip(displayed_source_axes,
                                            displayed_shape,
                                            displayed_scale)
                if ax in "ZYX" and ax_s > 1
            ]

            acquisition_root = save_zarr(
                output_filename,
                data=None,
                shape=acquisition_fun_shape,
                chunk_size=True,
                group_name="labels/acquisition_fun/0",
                dtype=np.float32
            )

            segmentation_root = save_zarr(
                output_filename,
                data=None,
                shape=acquisition_fun_shape,
                chunk_size=True,
                group_name="labels/segmentation/0",
                dtype=np.int32
            )

            image_group_metadata = image_group.metadata
            for layer_type in image_group.layer_type:
                layer_source_axes = image_group.source_axes[layer_type]
                image_group_metadata[layer_type]["roi"] = [
                    tuple(
                        self._roi.get(ax, slice(None))
                        for ax in layer_source_axes
                    )
                ]

                spaial_axes = "".join([
                    ax for ax in layer_source_axes
                    if ax in "ZYX"
                ])

                if "images" in layer_type:
                    if "C" in displayed_source_axes:
                        spaial_axes += "C"

                image_group_metadata[layer_type]["axes"] = spaial_axes

            acquisition_fun_max = self.compute_acquisition_fun(
                image_group_metadata,
                acquisition_fun=acquisition_root["labels/acquisition_fun/0"],
                segmentation_out=segmentation_root["labels/segmentation/0"],
                MC_repetitions=MC_repetitions,
                max_samples=max_samples,
                patch_size=patch_size
            )
            self.image_pb.setValue(n + 1)

            if output_filename:
                acquisition_root = output_filename
                segmentation_root = output_filename


            # Downsample the acquisition function
            acquisition_fun_ms = downsample_image(
                acquisition_root,
                source_axes="YX",
                data_group="labels/acquisition_fun/0",
                scale=4,
                num_scales=5,
                reference_source_axes=displayed_source_axes,
                reference_scale=displayed_scale
            )

            # Downsample the segmentation output
            segmentation_ms = downsample_image(
                segmentation_root,
                source_axes="YX",
                data_group="labels/segmentation/0",
                scale=4,
                num_scales=5,
                reference_source_axes=displayed_source_axes,
                reference_scale=displayed_scale,
            )

            acquisition_fun_ms_kwargs = dict(
                name=group_name + " acquisition function",
                multiscale=True,
                opacity=0.8,
                scale=acquisition_fun_scale,
                blending="translucent_no_depth",
                colormap="magma",
                contrast_limits=(0, acquisition_fun_max),
            )

            segmentation_ms_kwargs = dict(
                name=group_name + " acquisition function",
                multiscale=True,
                opacity=0.8,
                scale=acquisition_fun_scale,
                blending="translucent_no_depth",
            )

            if output_filename:
                acquisition_loader = viewer.open
                segmentation_loader = viewer.open

                acquisition_fun_ms_kwargs["layer_type"] = "image"
                segmentation_ms_kwargs["layer_type"] = "labels"

            else:
                acquisition_loader = viewer.add_image
                segmentation_loader = viewer.add_labels

            new_acquisition_layer = acquisition_loader(acquisition_fun_ms, **acquisition_fun_ms_kwargs)
            new_segmentation_layer = segmentation_loader(segmentation_ms, **segmentation_ms_kwargs)

            self.viewer.layers.selection.clear()
            self.viewer.layers.selection.add(new_acquisition_layer)
            self.viewer.layers.selection.add(new_segmentation_layer)

            image_groups_manager.update_group()


if __name__ == "__main__":
    import skimage
    viewer = napari.Viewer()

    img = skimage.data.astronaut()
    viewer.add_image(img[..., 0], blending="opaque", colormap="red")
    viewer.add_image(img[..., 1], blending="additive", colormap="green")
    viewer.add_image(img[..., 2], blending="additive", colormap="blue")

    image_groups_manager = ImageGroupsManager(viewer)
    viewer.window.add_dock_widget(image_groups_manager, area='right')

    labels_manager_widget = LabelsManager(viewer)
    acquisition_function = AcquisitionFunction(viewer, labels_manager_widget,
                                               image_groups_manager)
    viewer.window.add_dock_widget(acquisition_function, area='right')
    viewer.window.add_dock_widget(labels_manager_widget, area='right')

    napari.run()
