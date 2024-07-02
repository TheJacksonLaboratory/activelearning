from typing import List, Iterable, Union, Optional
import json

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QIntValidator
from qtpy.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
                            QLineEdit,
                            QComboBox,
                            QLabel,
                            QFileDialog,
                            QSpinBox,
                            QCheckBox,
                            QProgressBar,
                            QTreeWidget,
                            QTreeWidgetItem,
                            QAbstractItemView)

from functools import partial
from pathlib import Path
import torch
import napari
from napari.layers import Image, Labels, Layer
from napari.layers._multiscale_data import MultiScaleData
import tensorstore as ts
import numpy as np
import dask.array as da
import zarr

from ome_zarr.writer import write_multiscales_metadata, write_label_metadata

import dask.array as da
from models import train_cellpose
from datautils import get_dataloader

from cellpose import core, transforms
from cellpose.models import CellposeModel


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
    model = CellposeModel(gpu=False, model_type="cyto")
    model.mkldnn = False
    model.net.mkldnn = False

    if use_dropout:
        model.net.load_model(model.pretrained_model, device=model.device)
        model.net.eval()
        add_dropout(model.net)

    return model


def cellpose_probs(img, model):
    x = transforms.convert_image(img, channel_axis=2, channels=[0, 0])
    x = transforms.normalize_img(x, invert=False, axis=2)

    with torch.no_grad():
        y, _ = core.run_net(model.net, x)
        cellprob = torch.from_numpy(y[:, :, 2])
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


def get_next_name(name: str, group_names: Iterable[str]):
    if name in group_names:
        n_existing = sum(map(
            lambda exisitng_group_name:
            name in exisitng_group_name,
            group_names
        ))

        new_name = name + " (%i)" % n_existing
    else:
        new_name = name

    return new_name


def validate_child_name(group_names, previous_child_name, new_child_name):
    if previous_child_name in group_names:
        group_names.remove(previous_child_name)

    if new_child_name:
        new_child_name = get_next_name(new_child_name, group_names)
        group_names.add(new_child_name)

    return new_child_name


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


def save_zarr(output_filename, data, shape, chunk_size, name, dtype,
              is_multiscale: bool = False,
              metadata: Optional[dict] = None,
              is_label: bool = False):
    if not metadata:
        metadata = {}

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

    group_name = name
    if is_label:
        group_name = "labels/" + group_name

    if isinstance(data, MultiScaleData):
        data_ms = data
    else:
        data_ms = [data]

    num_scales = len(data_ms)

    if num_scales > 1:
        group_ms_names = [
            group_name + ("/%i" % s if is_multiscale else "")
            for s in range(num_scales)
        ]
    else:
        group_ms_names = [group_name + ("/0" if is_multiscale else "")]

    for data_ms_s, group_ms_s in zip(data_ms, group_ms_names):

        if data_ms_s is not None and not isinstance(data_ms_s, np.ndarray):
            data_ms_s = np.array(data_ms_s)

        out_grp.create_dataset(
            data=data_ms_s,
            name=group_ms_s,
            shape=shape if data_ms_s is None else data_ms_s.shape,
            chunks=chunks_size_axes,
            compressor=zarr.Blosc(clevel=9),
            write_empty_chunks=False,
            dtype=dtype if data_ms_s is None else data_ms_s.dtype,
            overwrite=True
        )

    if not isinstance(out_grp.store, zarr.MemoryStore):
        write_label_metadata(out_grp, group_name, **metadata)

    return out_grp


def downsample_image(z_root, source_axes, data_group, scale=4, num_scales=5,
                     reference_source_axes=None,
                     reference_scale=None,
                     reference_units=None):
    if isinstance(z_root, (Path, str)):
        source_arr = da.from_zarr(z_root, component=data_group)

        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': str(Path(z_root) / data_group),
            },
        }

        ts_array = ts.open(spec).result()
        z_ms = [ts_array]

    else:
        source_arr = da.from_zarr(z_root[data_group])
        z_ms = [source_arr]

    data_group = "/".join(data_group.split("/")[:-1])
    groups_root = data_group + "/%i"

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
        target_arr = target_arr.rechunk()

        if isinstance(z_root, (Path, str)):
            z_ms.append(z_ms[-1][downscale_selection])

            target_arr.to_zarr(z_root,
                               component=groups_root % s,
                               compressor=zarr.Blosc(clevel=9),
                               write_empty_chunks=False,
                               overwrite=True)

            source_arr = da.from_zarr(z_root, component=groups_root % s)

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

    return z_ms


def get_source_data(layer: Layer):
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
        data_group = str(Path(data_group) / "0")

    if not input_filename:
        input_filename = None

    if not data_group:
        data_group = None

    return input_filename, data_group


class LayerChannel(QTreeWidgetItem):
    def __init__(self, layer: Layer, channel: int = 0,
                 source_axes: str = "TZYX"):
        self.layer = layer
        self._channel = None
        self._source_axes = None
        self._source_data = None
        self._data_group = None

        super().__init__([layer.name])
        layer.events.name.connect(self._update_name)

        self.channel = channel
        self.source_axes = source_axes

    def _update_name(self, event):
        self.setText(0, self.layer.name)

    def _update_source_data(self):
        self._source_data, self._data_group = get_source_data(self.layer)
        if self._source_data is None:
            self._source_data = self.layer.data

    @property
    def source_data(self):
        if self._source_data is None:
            self._update_source_data()

        return self._source_data

    @source_data.setter
    def source_data(self, source_data):
        if isinstance(source_data, (Path, str)):
            source_data = str(source_data)
            self.layer._source = napari.layers._source.Source(
                path=source_data
            )

        self._source_data = source_data
        self.parent().updated = True

    @property
    def data_group(self):
        if not self._data_group:
            self._update_source_data()

        return self._data_group

    @data_group.setter
    def data_group(self, data_group):
        self._data_group = data_group
        self.parent().updated = True

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

    @property
    def scale(self):
        return self.layer.scale

    @scale.setter
    def scale(self, new_scale: Iterable[float]):
        self.layer.scale = new_scale

    @source_axes.setter
    def source_axes(self, source_axes: str):
        if "C" in source_axes:
            source_axes = list(source_axes)
            source_axes.remove("C")
            source_axes = "".join(source_axes)

        self._source_axes = source_axes
        self.setText(3, str(self._source_axes))

    @property
    def visible(self):
        return self.layer.visible

    @visible.setter
    def visible(self, visibility: bool):
        self.layer.visible = visibility

    @property
    def selected(self):
        viewer = napari.current_viewer()
        return self.layer in viewer.layers.selection

    @selected.setter
    def selected(self, is_selected: bool):
        viewer = napari.current_viewer()
        if is_selected:
            viewer.layers.selection.add(self.layer)
        else:
            viewer.layers.selection.remove(self.layer)


class LayersGroup(QTreeWidgetItem):
    def __init__(self, layers_group_name: str,
                 source_axes: Optional[str] = None,
                 use_as_input_image: Optional[bool] = False,
                 use_as_sampling_mask: Optional[bool] = False
                 ):

        self._layers_group_name = None
        self._use_as_input_image = False
        self._use_as_sampling_mask = False

        self._source_axes_no_channels = None
        self._source_axes = None
        self._source_data = None
        self._data_group = None

        super().__init__()

        self.layers_group_name = layers_group_name
        self.use_as_input_image = use_as_input_image
        self.use_as_sampling_mask = use_as_sampling_mask
        self.source_axes = source_axes

        self.updated = True

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
        if self.childCount():
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

        else:
            self._source_data = None
            self._data_group = None

        self.updated = False

    @property
    def source_data(self):
        if self._source_data is None or self.updated:
            self._update_source_data()

        return self._source_data

    @property
    def data_group(self):
        if not self._data_group or self.updated:
            self._update_source_data()

        return self._data_group

    @property
    def metadata(self):
        metadata = {
            "modality": self._layers_group_name,
            "filenames": self.source_data,
            "data_group": self.data_group,
            "source_axes": self._source_axes,
            "add_to_output": not self._use_as_sampling_mask
        }

        return metadata

    @property
    def layers_group_name(self):
        return self._layers_group_name

    @layers_group_name.setter
    def layers_group_name(self, layers_group_name: str):
        if self.parent():
            layers_group_name = validate_child_name(
                self.parent().layers_groups_names,
                self._layers_group_name,
                layers_group_name
            )

            if (self.parent().group_name is None
               or "unset" in self.parent().group_name):
                self.parent().group_name = layers_group_name

        self._layers_group_name = layers_group_name

        if self._layers_group_name:
            self.setText(0, self._layers_group_name)
        else:
            self.setText(0, "unset")

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

    @property
    def visible(self):
        is_visible = False
        for idx in range(self.childCount()):
            is_visible |= self.child(idx).visible

        return is_visible

    @visible.setter
    def visible(self, visibility: bool):
        for idx in range(self.childCount()):
            self.child(idx).visible = visibility

    @property
    def selected(self):
        is_selected = False
        for idx in range(self.childCount()):
            child = self.child(idx)
            if isinstance(child, LayerChannel):
                is_selected |= child.selected

        return is_selected

    @selected.setter
    def selected(self, is_selected: bool):
        for idx in range(self.childCount()):
            child = self.child(idx)
            if isinstance(child, LayerChannel):
                child.selected = is_selected

    @property
    def use_as_input_image(self):
        return self._use_as_input_image

    @use_as_input_image.setter
    def use_as_input_image(self, use_it: bool):
        self._use_as_input_image = use_it

        use_as = []
        if self.use_as_input_image:
            use_as.append("Input")

        if self.use_as_sampling_mask:
            use_as.append("Sampling mask")

        self.setText(1, "/".join(use_as))

    @property
    def use_as_sampling_mask(self):
        return self._use_as_sampling_mask

    @use_as_sampling_mask.setter
    def use_as_sampling_mask(self, use_it: bool):
        self._use_as_sampling_mask = use_it

        use_as = []
        if self.use_as_input_image:
            use_as.append("Input")

        if self.use_as_sampling_mask:
            use_as.append("Sampling mask")

        self.setText(1, "/".join(use_as))

    def add_layer(self, layer: Layer, channel: Optional[int] = None,
                  source_axes: Optional[str] = None):
        if channel is None:
            channel = self.childCount()

        if source_axes is None:
            source_axes = self._source_axes_no_channels

        if not self._layers_group_name:
            self.layers_group_name = get_basename(layer.name)

        self.updated = True

        new_layer_channel = LayerChannel(layer, channel=channel,
                                         source_axes=source_axes)

        self.addChild(new_layer_channel)

        new_layer_channel.setExpanded(True)

        self.source_axes = source_axes

        return new_layer_channel

    def remove_layer(self, layer_channel: LayerChannel):
        removed_channel = layer_channel.channel

        for idx in range(self.childCount()):
            curr_layer_channel = self.child(idx)

            if curr_layer_channel.channel > removed_channel:
                curr_layer_channel.channel = curr_layer_channel.channel - 1

        self._update_source_axes()
        self.updated = True

    def takeChild(self, index: int):
        child = super(LayersGroup, self).takeChild(index)
        self.remove_layer(child)
        return child

    def removeChild(self, child: QTreeWidgetItem):
        self.remove_layer(child)
        super(LayersGroup, self).removeChild(child)

    def addChild(self, child: QTreeWidgetItem):
        if isinstance(child, LayerChannel):
            self.parent().parent().add_managed_layer(child.layer, child)

        super(LayersGroup, self).addChild(child)

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

    def save_group(self, output_dir: Path, metadata: Optional[dict] = None):
        if self.parent():
            group_name = self.parent().group_name
        else:
            group_name = "unset"

        output_filename = output_dir / (group_name + ".zarr")

        source_data = self.source_data

        name = get_basename(self.layers_group_name)

        is_multiscale = isinstance(source_data, MultiScaleData)
        is_label = not self._use_as_input_image

        save_zarr(
            output_filename,
            data=source_data,
            shape=source_data.shape,
            chunk_size=True,
            name=name,
            dtype=source_data.dtype,
            metadata=metadata,
            is_label=is_label,
            is_multiscale=is_multiscale
        )

        for idx in range(self.childCount()):
            self.child(idx).source_data = str(output_filename)
            self.child(idx).data_group = (("labels/" if is_label else "")
                                          + name
                                          + ("/0" if is_multiscale else ""))

        self.updated = True


class ImageGroup(QTreeWidgetItem):
    def __init__(self, group_name: Optional[str] = None,
                 group_dir: Optional[Union[Path, str]] = None):
        self._group_metadata = {}
        self._group_name = None
        self._group_dir = None

        self.layers_groups_names = set()

        super().__init__()

        self.group_name = group_name
        self.group_dir = group_dir

    @property
    def group_name(self):
        return self._group_name

    @group_name.setter
    def group_name(self, group_name: str):
        if self.parent():
            group_name = validate_child_name(
                self.parent().group_names,
                self._group_name,
                group_name
            )

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

    @property
    def metadata(self):
        self.update_group_metadata()

        return self._group_metadata

    @property
    def visible(self):
        is_visible = False
        for idx in range(self.childCount()):
            is_visible |= self.child(idx).visible

        return is_visible

    @visible.setter
    def visible(self, visibility: bool):
        for idx in range(self.childCount()):
            self.child(idx).visible = visibility

    @property
    def selected(self):
        is_selected = False
        for idx in range(self.childCount()):
            child = self.child(idx)
            if isinstance(child, LayersGroup):
                is_selected |= child.selected

        return is_selected

    @selected.setter
    def selected(self, is_selected: bool):
        for idx in range(self.childCount()):
            child = self.child(idx)
            if isinstance(child, LayersGroup):
                self.child(idx).selected = is_selected

    @property
    def input_layers_group(self):
        layers_group_index = list(filter(
            lambda idx:
            self.child(idx).use_as_input_image,
            range(self.childCount())
        ))

        if layers_group_index:
            layers_group_index = layers_group_index[0]

        else:
            layers_group_index = None

        return layers_group_index

    @input_layers_group.setter
    def input_layers_group(self, input_idx: Union[int, None]):
        for idx in range(self.childCount()):
            self.child(idx).use_as_input_image = False

        if input_idx is not None:
            self.child(input_idx).use_as_input_image = True

    @property
    def sampling_mask_layers_group(self):
        layers_group_index = list(filter(
            lambda idx:
            self.child(idx).use_as_sampling_mask,
            range(self.childCount())
        ))

        if layers_group_index:
            layers_group_index = layers_group_index[0]

        else:
            layers_group_index = None

        return layers_group_index

    @sampling_mask_layers_group.setter
    def sampling_mask_layers_group(self, sampling_mask_idx: Union[int, None]):
        for idx in range(self.childCount()):
            self.child(idx).use_as_sampling_mask = False

        if sampling_mask_idx is not None:
            self.child(sampling_mask_idx).use_as_sampling_mask = True

    def getLayersGroup(self, layers_group_name: str):
        layers_group = list(filter(
            lambda layers_group:
            layers_group.layers_group_name == layers_group_name,
            map(lambda idx: self.child(idx), range(self.childCount()))
        ))

        if layers_group:
            layers_group = layers_group[0]
        else:
            layers_group = None

        return layers_group

    def add_layers_group(self, layers_group_name: Optional[str] = None,
                         source_axes: Optional[str] = None,
                         use_as_input_image: Optional[bool] = None,
                         use_as_sampling_mask: Optional[bool] = False,
                         ):
        if use_as_input_image is None:
            use_as_input_image = self.input_layers_group is None

        if layers_group_name is None:
            if use_as_input_image:
                layers_group_name = "images"
            elif use_as_sampling_mask:
                layers_group_name = "masks"

        new_layers_group = LayersGroup(
            layers_group_name,
            source_axes=source_axes,
            use_as_input_image=use_as_input_image,
            use_as_sampling_mask=use_as_sampling_mask
        )

        self.addChild(new_layers_group)

        new_layers_group.setExpanded(True)

        return new_layers_group

    def move_layer(self, src_layers_group: LayersGroup,
                   dst_layers_group: Optional[Union[str, LayersGroup]] = None,
                   layer_channel: Optional[LayerChannel] = None):
        if layer_channel:
            layer_channel_list = [
                src_layers_group.takeChild(
                    src_layers_group.indexOfChild(layer_channel)
                )
            ]

        else:
            layer_channel_list = src_layers_group.takeChildren()

        if dst_layers_group is not None:
            if isinstance(dst_layers_group, LayersGroup):
                dst_layers_group = dst_layers_group

            else:
                dst_layers_group_name = dst_layers_group
                dst_layers_group = self.getLayersGroup(dst_layers_group_name)

                if not dst_layers_group:
                    dst_layers_group = self.add_layers_group(
                        dst_layers_group_name
                    )

            for curr_layer_channel in layer_channel_list:
                dst_layers_group.add_layer(
                    curr_layer_channel.layer,
                    channel=None,
                    source_axes=curr_layer_channel.source_axes
                )

        if not src_layers_group.childCount():
            self.removeChild(src_layers_group)

        return dst_layers_group

    def update_group_metadata(self):
        self._group_metadata = {}

        for layers_group in map(lambda idx: self.child(idx),
                                range(self.childCount())):
            self._group_metadata[layers_group.layers_group_name] =\
                 layers_group.metadata


class ImageGroupRoot(QTreeWidgetItem):
    def __init__(self):
        super().__init__(["Image groups"])
        self.managed_layers = {}
        self.group_names = set()

    def addChild(self, child: QTreeWidgetItem):
        if isinstance(child, ImageGroup):
            if not child.group_name:
                group_name = "unset"

            group_name = validate_child_name(self.group_names, "", group_name)

            child.group_name = group_name

        super(ImageGroupRoot, self).addChild(child)

    def removeChild(self, child: QTreeWidgetItem):
        if (isinstance(child, ImageGroup)
           and child.group_name in self.group_names):
            self.group_names.remove(child.group_name)

        super(ImageGroupRoot, self).removeChild(child)

    def takeChild(self, index: int):
        child = super(ImageGroupRoot, self).takeChild(index)

        if isinstance(child, ImageGroup):
            self.group_names.remove(child.group_name)

        return child

    def takeChildred(self):
        children = super(ImageGroupRoot, self).takeChildren()
        for child in children:
            if isinstance(child, ImageGroup):
                self.group_names.remove(child.group_name)

        return children

    def add_managed_layer(self, layer: Layer, layer_channel: LayerChannel):
        if layer not in self.managed_layers:
            self.managed_layers[layer] = []

        self.managed_layers[layer].append(layer_channel)

    def remove_managed_layer(self, event):
        removed_layer = event.value

        layer_channel_list = self.managed_layers.get(removed_layer, [])
        for layer_channel in layer_channel_list:
            layers_group = layer_channel.parent()
            image_group = layers_group.parent()

            layers_group.removeChild(layer_channel)

            if not layers_group.childCount():
                image_group.removeChild(layers_group)

            if not image_group.childCount():
                self.removeChild(image_group)

        if layer_channel_list:
            self.managed_layers.pop(removed_layer)

        self.setSelected(True)


class ImageGroupEditor(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.viewer = viewer

        self._active_image_group: Union[None, ImageGroup] = None
        self._active_layers_group: Union[None, LayersGroup] = None
        self._active_layer_channel: Union[None, LayerChannel] = None

        self.group_name_lbl = QLabel("Group name:")
        self.group_name_le = QLineEdit("None selected")
        self.group_name_le.setEnabled(False)

        self.layer_name_lbl = QLabel("Channel name:")
        self.display_name_lbl = QLabel("None selected")

        self.layers_group_name_lbl = QLabel("Channels group name:")
        self.layers_group_name_cmb = QComboBox()
        self.layers_group_name_cmb.setEditable(True)
        self.layers_group_name_cmb.setEnabled(False)

        self.layer_axes_lbl = QLabel("Axes order:")
        self.edit_axes_le = QLineEdit("None selected")
        self.edit_axes_le.setEnabled(False)

        self.use_as_input_chk = QCheckBox("Use as input")
        self.use_as_input_chk.setEnabled(False)
        self.use_as_sampling_chk = QCheckBox("Use as sampling mask")
        self.use_as_sampling_chk.setEnabled(False)

        self.layer_channel_lbl = QLabel("Channel:")
        self.edit_channel_spn = QSpinBox(minimum=0, maximum=0)
        self.edit_channel_spn.setEnabled(False)

        self.output_dir_lbl = QLabel("Output directory:")
        self.output_dir_le = QLineEdit("Unset")
        self.output_dir_dlg = QFileDialog(fileMode=QFileDialog.Directory)
        self.output_dir_btn = QPushButton("...")
        self.output_dir_le.setEnabled(False)
        self.output_dir_btn.setEnabled(False)

        self.output_dir_btn.clicked.connect(self.output_dir_dlg.show)
        self.output_dir_dlg.directoryEntered.connect(self.update_output_dir)

        self.layers_group_name_cmb.lineEdit().returnPressed.connect(
            self.update_layer_type
        )
        self.group_name_le.returnPressed.connect(self.update_group_name)
        self.output_dir_le.returnPressed.connect(self.update_output_dir)

        self.edit_axes_le.returnPressed.connect(self.update_source_axes)
        self.use_as_input_chk.toggled.connect(
            self.update_use_as_input
        )
        self.use_as_sampling_chk.toggled.connect(
            self.update_use_as_sampling
        )

        self.edit_channel_spn.editingFinished.connect(self.update_channels)
        self.edit_channel_spn.valueChanged.connect(self.update_channels)

        self.edit_1_lyt = QHBoxLayout()
        self.edit_1_lyt.addWidget(self.group_name_lbl)
        self.edit_1_lyt.addWidget(self.group_name_le)

        self.edit_2_lyt = QHBoxLayout()
        self.edit_2_lyt.addWidget(self.layer_name_lbl)
        self.edit_2_lyt.addWidget(self.display_name_lbl)
        self.edit_2_lyt.addWidget(self.layers_group_name_lbl)
        self.edit_2_lyt.addWidget(self.layers_group_name_cmb)

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
        self.edit_5_lyt.addWidget(self.use_as_input_chk)
        self.edit_5_lyt.addWidget(self.use_as_sampling_chk)

        self.edit_lyt = QVBoxLayout()
        self.edit_lyt.addLayout(self.edit_1_lyt)
        self.edit_lyt.addLayout(self.edit_2_lyt)
        self.edit_lyt.addLayout(self.edit_3_lyt)
        self.edit_lyt.addLayout(self.edit_4_lyt)
        self.edit_lyt.addLayout(self.edit_5_lyt)

        self.setLayout(self.edit_lyt)

    def _clear_image_group(self):
        self.layers_group_name_cmb.clear()
        self.group_name_le.setText("None selected")

        self.group_name_le.setEnabled(False)
        self.output_dir_le.setEnabled(False)
        self.output_dir_btn.setEnabled(False)

    def _clear_layers_group(self):
        self.edit_axes_le.setText("None selected")
        self.layers_group_name_cmb.setEnabled(False)
        self.edit_axes_le.setEnabled(False)
        self.use_as_input_chk.setEnabled(False)
        self.use_as_sampling_chk.setEnabled(False)

    def _clear_layer_channel(self):
        self.edit_channel_spn.setValue(0)
        self.edit_channel_spn.setMaximum(0)
        self.edit_channel_spn.setEnabled(False)

    def _fill_image_group(self):
        self._clear_image_group()

        if self._active_image_group:
            self.output_dir_le.setText(str(self._active_image_group.group_dir))
            self.group_name_le.setText(self._active_image_group.group_name)

            self.layers_group_name_cmb.clear()
            for idx in range(self._active_image_group.childCount()):
                layers_group = self._active_image_group.child(idx)
                self.layers_group_name_cmb.addItem(
                    layers_group.layers_group_name
                )

            self.output_dir_btn.setEnabled(True)
            self.output_dir_le.setEnabled(True)
            self.group_name_le.setEnabled(True)

        self._fill_layers_group()

    def _fill_layers_group(self):
        self._clear_layers_group()

        if self._active_layers_group:
            self.layers_group_name_cmb.lineEdit().setText(
                self._active_layers_group.layers_group_name
            )
            self.edit_axes_le.setText(
                self._active_layers_group.source_axes
            )

            self.use_as_input_chk.setChecked(
                self._active_layers_group.use_as_input_image
            )
            self.use_as_sampling_chk.setChecked(
                self._active_layers_group.use_as_sampling_mask
            )

            self.layers_group_name_cmb.setEnabled(True)
            self.edit_axes_le.setEnabled(True)
            self.use_as_input_chk.setEnabled(True)
            self.use_as_sampling_chk.setEnabled(True)

        self._fill_layer()

    def _fill_layer(self):
        self._clear_layer_channel()

        if self._active_layer_channel:
            self.display_name_lbl.setText(
                self._active_layer_channel.layer.name
            )
            self.edit_axes_le.setText(self._active_layer_channel.source_axes)
            self.edit_axes_le.setEnabled(True)

            self.edit_channel_spn.setMaximum(
                self._active_layers_group.childCount() - 1
            )
            self.edit_channel_spn.setValue(self._active_layer_channel.channel)
            self.edit_channel_spn.setEnabled(True)

    def update_output_dir(self, output_dir: Optional[Union[Path, str]] = None):
        if output_dir:
            self.output_dir_le.setText(str(output_dir))
        else:
            self.output_dir_le.setText(str(self._active_image_group.group_dir))

        if not self._active_image_group:
            return

        new_output_dir = self.output_dir_le.text()

        if new_output_dir.lower() in ("unset", "none", ""):
            new_output_dir = None

        if self._active_image_group.group_dir != new_output_dir:
            self._active_image_group.group_dir = new_output_dir

    def update_group_name(self, name: Optional[str] = None):
        if not self._active_image_group:
            return

        name = self.group_name_le.text()

        if self._active_image_group.group_name != name:
            self._active_image_group.group_name = name

    def update_channels(self, channel: Optional[int] = None):
        if not self._active_layer_channel:
            return

        if not self._active_layers_group:
            return

        if channel:
            self.edit_channel_spn.setValue(channel)
        else:
            channel = self.edit_channel_spn.value()

        prev_channel = self._active_layer_channel.channel
        if prev_channel != channel:
            self._active_layers_group.move_channel(prev_channel, channel)

    def update_source_axes(self, source_axes: Optional[str] = None):
        if not self._active_layers_group and not self._active_layer_channel:
            return

        if source_axes:
            self.edit_axes_le.setText(source_axes)
        else:
            source_axes = self.edit_axes_le.text().upper()

        if self._active_layers_group:
            if self._active_layers_group.source_axes != source_axes:
                self._active_layers_group.source_axes = source_axes

        if self._active_layer_channel:
            if self._active_layer_channel.source_axes != source_axes:
                self._active_layer_channel.source_axes = source_axes

        display_source_axes = list(source_axes.lower())
        if "c" in display_source_axes:
            display_source_axes.remove("c")
        display_source_axes = tuple(display_source_axes)

        if display_source_axes != self.viewer.dims.axis_labels:
            self.viewer.dims.axis_labels = display_source_axes

    def update_layer_type(self, layers_group_name: Optional[str] = None):
        if not self._active_layers_group or not self._active_image_group:
            return

        if layers_group_name:
            self.layers_group_name_cmb.lineEdit().setText(layers_group_name)
        else:
            layers_group_name = self.layers_group_name_cmb.lineEdit().text()

        if self._active_layers_group.layers_group_name != layers_group_name:

            dst_layers_group = self._active_image_group.getLayersGroup(
                layers_group_name
            )

            if dst_layers_group:
                self._active_layers_group =\
                    self._active_image_group.move_layer(
                        self._active_layers_group,
                        dst_layers_group,
                        layer_channel=self._active_layer_channel
                    )
            else:
                self._active_layers_group.layers_group_name = layers_group_name

            self._active_layer_channel = None
            self._clear_layer_channel()

    def update_use_as_input(self, use_it: Optional[bool] = None):
        if not self._active_layers_group:
            return

        if use_it is not None:
            self.use_as_input_chk.setChecked(use_it)
        else:
            use_it = self.use_as_input_chk.isChecked()

        layers_group_idx = self.active_image_group.indexOfChild(
            self._active_layers_group
        )

        if use_it:
            self._active_image_group.input_layers_group = layers_group_idx

        elif self._active_image_group.input_layers_group == layers_group_idx:
            self._active_image_group.input_layers_group = None

    def update_use_as_sampling(self, use_it: Optional[bool] = None):
        if not self._active_layers_group:
            return

        if use_it is not None:
            self.use_as_sampling_chk.setChecked(use_it)
        else:
            use_it = self.use_as_sampling_chk.isChecked()

        layers_group_idx = self.active_image_group.indexOfChild(
            self._active_layers_group
        )

        if use_it:
            self._active_image_group.sampling_mask_layers_group =\
                 layers_group_idx

        elif (self._active_image_group.sampling_mask_layers_group
              == layers_group_idx):
            self._active_image_group.sampling_mask_layers_group = None

    @property
    def active_image_group(self):
        return self._active_image_group

    @active_image_group.setter
    def active_image_group(self,
                           active_image_group: Optional[ImageGroup] = None):
        self._active_image_group = active_image_group
        self._active_layers_group = None
        self._active_layer_channel = None

        self._fill_image_group()

    @property
    def active_layers_group(self):
        return self._active_layers_group

    @active_layers_group.setter
    def active_layers_group(self,
                            active_layers_group: Optional[LayersGroup] = None):
        self._active_layers_group = active_layers_group
        self._active_layer_channel = None

        self._fill_layers_group()

    @property
    def active_layer_channel(self):
        return self._active_layer_channel

    @active_layer_channel.setter
    def active_layer_channel(
       self,
       active_layer_channel: Optional[LayerChannel] = None
       ):
        self._active_layer_channel = active_layer_channel
        self._fill_layer()


class ImageGroupsManager(QWidget):

    layer_selected = Signal(QTreeWidgetItem)

    def __init__(self, viewer: napari.Viewer,
                 default_axis_labels: str = "TZYX"):
        super().__init__()

        self.viewer = viewer

        ndims = self.viewer.dims.ndisplay
        extra_dims = ndims - len(default_axis_labels)
        if extra_dims <= 0:
            axis_labels = default_axis_labels[extra_dims:]
        else:
            axis_labels = ("".join(map(str, range(len(default_axis_labels),
                                                  ndims)))
                           + default_axis_labels)

        self.viewer.dims.axis_labels = list(axis_labels)

        self._active_layer_channel: Union[None, LayerChannel] = None
        self._active_layers_group: Union[None, LayersGroup] = None
        self._active_image_group: Union[None, ImageGroup] = None

        self.image_groups_tw = QTreeWidget()

        self.image_groups_tw.setColumnCount(5)
        self.image_groups_tw.setHeaderLabels([
            "Group name",
            "Use",
            "Channels",
            "Axes order",
            "Output directory",
        ])
        self.image_groups_tw.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )

        self.image_groups_tw.itemDoubleClicked.connect(
            self._focus_active_element
        )
        self.image_groups_tw.itemSelectionChanged.connect(
            self._get_active_item
        )
        self.image_groups_tw.setExpandsOnDoubleClick(False)       

        self.groups_root = ImageGroupRoot()
        viewer.layers.events.removed.connect(
            self.groups_root.remove_managed_layer
        )
        self.image_groups_tw.addTopLevelItem(self.groups_root)
        self.groups_root.setExpanded(True)

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

        self.new_layers_group_btn = QPushButton("New layers group")
        self.new_layers_group_btn.setEnabled(False)

        self.remove_layers_group_btn = QPushButton("Remove layers group")
        self.remove_layers_group_btn.setEnabled(False)

        self.save_layers_group_btn = QPushButton("Save layers group")
        self.save_layers_group_btn.setEnabled(False)

        self.add_layer_btn = QPushButton("Add layer(s)")
        self.add_layer_btn.setEnabled(False)

        self.remove_layer_btn = QPushButton("Remove layer")
        self.remove_layer_btn.setEnabled(False)

        self.save_metadata_btn = QPushButton("Save metadata")
        self.save_metadata_btn.setEnabled(False)

        self.new_layers_group_btn.clicked.connect(self.create_layers_group)
        self.remove_layers_group_btn.clicked.connect(self.remove_layers_group)
        self.save_layers_group_btn.clicked.connect(self.save_layers_group)
        self.add_layer_btn.clicked.connect(self.add_layers_to_group)
        self.remove_layer_btn.clicked.connect(self.remove_layer)
        self.save_metadata_btn.clicked.connect(self.dump_dataset_specs)

        self.group_buttons_lyt = QHBoxLayout()
        self.group_buttons_lyt.addWidget(self.new_group_btn)
        self.group_buttons_lyt.addWidget(self.add_group_btn)
        self.group_buttons_lyt.addWidget(self.remove_group_btn)

        self.layers_group_buttons_lyt = QHBoxLayout()
        self.layers_group_buttons_lyt.addWidget(self.new_layers_group_btn)
        self.layers_group_buttons_lyt.addWidget(self.remove_layers_group_btn)
        self.layers_group_buttons_lyt.addWidget(self.save_layers_group_btn)

        self.layer_buttons_lyt = QHBoxLayout()
        self.layer_buttons_lyt.addWidget(self.add_layer_btn)
        self.layer_buttons_lyt.addWidget(self.remove_layer_btn)
        self.layer_buttons_lyt.addWidget(self.save_metadata_btn)

        self.show_editor_chk = QCheckBox("Edit group properties")
        self.show_editor_chk.setChecked(False)
        self.show_editor_chk.toggled.connect(self._show_editor)

        self.layers_editor = ImageGroupEditor(viewer)
        self.layers_editor.setVisible(False)
        self.mask_creator = MaskGenerator(viewer)
        self.mask_creator.setVisible(False)

        self.group_lyt = QVBoxLayout()
        self.group_lyt.addLayout(self.group_buttons_lyt)
        self.group_lyt.addLayout(self.layers_group_buttons_lyt)
        self.group_lyt.addLayout(self.layer_buttons_lyt)
        self.group_lyt.addWidget(self.show_editor_chk)
        self.group_lyt.addWidget(self.layers_editor)
        self.group_lyt.addWidget(self.mask_creator)
        self.group_lyt.addWidget(self.image_groups_tw)

        self.setLayout(self.group_lyt)

    def _show_editor(self, show: bool = False):
        self.layers_editor.setVisible(show)
        self.mask_creator.setVisible(show)

    def _get_active_item(self):
        item = self.image_groups_tw.selectedItems()
        if isinstance(item, list):
            if not len(item):
                return

            item = item[-1]

        self._active_layer_channel = None
        self._active_layers_group = None
        self._active_image_group = None

        if isinstance(item, LayerChannel):
            self._active_layer_channel = item

        elif isinstance(item, LayersGroup):
            self._active_layers_group = item

        elif isinstance(item, ImageGroup) and item != self.groups_root:
            self._active_image_group = item

        if self._active_layer_channel:
            self._active_layers_group = self._active_layer_channel.parent()

        if self._active_layers_group:
            self._active_image_group = self._active_layers_group.parent()

        self.mask_creator.active_image_group = self._active_image_group
        self.mask_creator.active_layers_group = self._active_layers_group

        self.layers_editor.active_image_group = self._active_image_group
        self.layers_editor.active_layers_group = self._active_layers_group
        self.layers_editor.active_layer_channel = self._active_layer_channel

        self.remove_group_btn.setEnabled(self._active_image_group is not None)
        self.add_group_btn.setEnabled(self._active_image_group is not None)
        self.new_layers_group_btn.setEnabled(
            self._active_image_group is not None
        )

        self.remove_layers_group_btn.setEnabled(
            self._active_layers_group is not None
        )

        self.add_layer_btn.setEnabled(self._active_layers_group is not None)

        self.remove_layer_btn.setEnabled(
            self._active_layer_channel is not None
        )

        self.save_layers_group_btn.setEnabled(
            self._active_layers_group is not None
            and isinstance(self._active_image_group.group_dir, (str, Path))
            and not isinstance(self._active_layers_group.source_data,
                               (str, Path))
        )

        self.save_metadata_btn.setEnabled(
            self._active_image_group is not None
            and isinstance(self._active_image_group.group_dir, (str, Path))
        )

        self.layer_selected.emit(item)

    def _focus_active_element(self):
        item = self.image_groups_tw.selectedItems()
        if isinstance(item, list):
            if not len(item):
                return

            item = item[0]

        if not item:
            return

        self.viewer.layers.selection.clear()
        for layer in self.viewer.layers:
            layer.visible = False

        if isinstance(item, (LayerChannel, LayersGroup, ImageGroup)):
            item.visible = True
            item.selected = True

            item.setExpanded(True)

    def update_group(self):
        selected_layers = self.viewer.layers.selection

        if not selected_layers or not self._active_image_group:
            return

        image_layers = set(filter(
            lambda layer: isinstance(layer, Image),
            selected_layers
        ))

        labels_layers = set(filter(
            lambda layer: isinstance(layer, Labels),
            selected_layers
        ))

        remaining_layers = set(filter(
            lambda layer: not isinstance(layer, (Image, Labels)),
            selected_layers
        ))

        for layers_type, layers_set in [("images", image_layers),
                                        ("masks", labels_layers),
                                        ("unset", remaining_layers)]:

            if not layers_set:
                continue

            self._active_layers_group =\
                self._active_image_group.getLayersGroup(layers_type)

            if self._active_layers_group is None:
                self.create_layers_group()

            self.viewer.layers.selection.clear()
            self.viewer.layers.selection = self.viewer.layers.selection.union(
                layers_set
            )

            self.add_layers_to_group()

        self._active_image_group.setExpanded(True)
        self.image_groups_tw.clearSelection()
        self._active_image_group.setSelected(True)

    def create_group(self):
        self._active_image_group = ImageGroup()
        self.groups_root.addChild(self._active_image_group)

        self._active_image_group.setExpanded(True)
        self.image_groups_tw.clearSelection()
        self._active_image_group.setSelected(True)

        self.update_group()

    def create_layers_group(self):
        if self._active_image_group is None:
            return

        active_source_axes = "".join(self.viewer.dims.axis_labels).upper()

        self._active_layers_group = self._active_image_group.add_layers_group(
            source_axes=active_source_axes
        )

        self._active_layers_group.setExpanded(True)
        self.image_groups_tw.clearSelection()
        self._active_layers_group.setSelected(True)

    def add_layers_to_group(self):
        selected_layers = sorted(
            map(lambda layer: (layer.name, layer),
                self.viewer.layers.selection)
        )

        selected_layers = list(zip(*selected_layers))[1]

        if not selected_layers:
            return

        for layer in selected_layers:
            self._active_layer_channel = self._active_layers_group.add_layer(
                layer=layer
            )

            self._active_layer_channel.setExpanded(True)

            self.image_groups_tw.clearSelection()
            self._active_layer_channel.setSelected(True)

        else:
            if (self._active_image_group.group_name is None
               or "unset" in self._active_image_group.group_name):
                self._active_image_group.group_name =\
                    self._active_layer_channel.layer.name

    def remove_layer(self):
        self._active_layers_group.removeChild(self._active_layer_channel)
        self._active_layer_channel = None

        self.image_groups_tw.clearSelection()
        self._active_layers_group.setSelected(True)

    def remove_layers_group(self):
        self._active_image_group.removeChild(self._active_layers_group)

        self._active_layer_channel = None
        self._active_layers_group = None

        self.image_groups_tw.clearSelection()
        self._active_image_group.setSelected(True)

    def remove_group(self):
        self.groups_root.removeChild(self._active_image_group)

        self._active_layer_channel = None
        self._active_layers_group = None
        self._active_image_group = None

        self.image_groups_tw.clearSelection()
        self.groups_root.setSelected(True)

    def save_layers_group(self):
        if not self._active_layers_group:
            return

        self._active_layers_group.save_group(
            output_dir=self._active_image_group.group_dir
        )

    def dump_dataset_specs(self):
        if self._active_image_group.group_dir:
            with open(self._active_image_group.group_dir
                      / (self._active_image_group.group_name
                         + "_metadata.json"), "w") as fp:
                fp.write(json.dumps(self._active_image_group.metadata))


class LabelItem(QTreeWidgetItem):
    _position = None
    _center = None
    _acquisition_val = None

    def __init__(self, acquisition_val: float, position: List[slice]):
        super().__init__()

        self.position = position
        self.acquisition_val = acquisition_val

    @property
    def position(self):
        return self._position

    @property
    def center(self):
        return self._center

    @position.setter
    def position(self, new_position: Iterable[slice]):
        self._position = new_position

        top_left, bottom_right = list(zip(*map(
            lambda ax_roi:
            (ax_roi.start, ax_roi.stop),
            new_position
        )))

        self._center = list(map(lambda tl, br: (tl + br) / 2,
                                top_left,
                                bottom_right))

        self.setText(1, "(" + ", ".join(map(str, self._center)) + ")")
        self.setText(2, "(" + ", ".join(map(str, top_left)) + ")")
        self.setText(3, "(" + ", ".join(map(str, bottom_right)) + ")")

    @property
    def acquisition_val(self):
        return self._acquisition_val

    @acquisition_val.setter
    def acquisition_val(self, new_acquisition_val: float):
        self._acquisition_val = new_acquisition_val
        self.setText(0, str(self._acquisition_val))


class LabelGroup(QTreeWidgetItem):
    _layer_channel = None

    def __init__(self, layer_channel: Optional[LayerChannel] = None):
        super().__init__()

        self.layer_channel = layer_channel

    @property
    def layer_channel(self):
        return self._layer_channel

    @layer_channel.setter
    def layer_channel(self, layer_channel: LayerChannel):
        self._layer_channel = layer_channel

        if (self._layer_channel.parent()
           and self._layer_channel.parent().parent()):
            self.setText(0, "Labels of group: "
                            + self._layer_channel.parent().parent().group_name)


class LabelGroupRoot(QTreeWidgetItem):
    def __init__(self):
        super().__init__(["Labeled groups"])
        self.managed_layers = {}

    def remove_managed_group(self, layer: Layer, label_group: LabelGroup):
        if (layer in self.managed_layers
           and label_group in self.managed_layers[layer]):
            self.managed_layers[layer].remove(label_group)

            if not self.managed_layers[layer]:
                self.managed_layers.pop(layer)

    def add_managed_layer(self, layer: Layer, label_group: LabelGroup):
        if layer not in self.managed_layers:
            self.managed_layers[layer] = []

        self.managed_layers[layer].append(label_group)

    def remove_managed_layer(self, event):
        removed_layer = event.value

        label_group_list = self.managed_layers.get(removed_layer, [])
        for label_group in label_group_list:
            self.removeChild(label_group)

    def addChild(self, child: QTreeWidgetItem):
        if isinstance(child, LabelGroup):
            if child.layer_channel:
                self.add_managed_layer(child.layer_channel.layer, child)

        super(LabelGroupRoot, self).addChild(child)

    def addChildren(self, children: Iterable[QTreeWidgetItem]):
        for child in children:
            if isinstance(child, LabelGroup):
                if child.layer_channel:
                    self.add_managed_label(child.layer_channel.layer, child)

        super(LabelGroupRoot, self).addChildred(children)

    def removeChild(self, child: QTreeWidgetItem):
        if isinstance(child, LabelGroup) and child.layer_channel:
            self.remove_managed_group(child.layer_channel.layer, child)

        super(LabelGroupRoot, self).removeChild(child)

    def takeChild(self, index: int):
        child = super(LabelGroupRoot, self).takeChild(index)
        if isinstance(child, LabelGroup) and child.layer_channel:
            self.remove_managed_group(child.layer_channel.layer, child)

        return child

    def takeChildren(self):
        children = super(LabelGroupRoot, self).takeChildren()
        for child in children:
            if isinstance(child, LabelGroup) and child.layer_channel:
                self.remove_managed_group(child.layer_channel.layer, child)

        return children


class LabelsManager(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.viewer = viewer

        self.labels_table_tw = QTreeWidget()
        self.labels_table_tw.setColumnCount(4)
        self.labels_table_tw.setHeaderLabels([
            "Acquisition value",
            "Sampling center",
            "Sampling top-left",
            "Sampling bottom-right"
        ])
        self.labels_table_tw.itemSelectionChanged.connect(self.focus_region)

        self.labels_group_root = LabelGroupRoot()
        self.labels_table_tw.addTopLevelItem(self.labels_group_root)

        self.labels_group_root.setExpanded(True)

        self.viewer.layers.events.removed.connect(
            self.labels_group_root.remove_managed_layer
        )

        self.prev_img_btn = QPushButton('<<')
        self.prev_img_btn.setEnabled(False)
        self.prev_img_btn.clicked.connect(partial(
            self.navigate, delta_image_index=-1
        ))

        self.prev_patch_btn = QPushButton('<')
        self.prev_patch_btn.setEnabled(False)
        self.prev_patch_btn.clicked.connect(partial(
            self.navigate, delta_patch_index=-1
        ))

        self.next_patch_btn = QPushButton('>')
        self.next_patch_btn.setEnabled(False)
        self.next_patch_btn.clicked.connect(partial(
            self.navigate, delta_patch_index=1
        ))

        self.next_img_btn = QPushButton('>>')
        self.next_img_btn.setEnabled(False)
        self.next_img_btn.clicked.connect(partial(
            self.navigate, delta_image_index=1
        ))

        self.edit_labels_btn = QPushButton("Edit current labels")
        self.edit_labels_btn.setEnabled(False)
        self.edit_labels_btn.clicked.connect(self.edit_labels)

        self.commit_btn = QPushButton("Commit changes")
        self.commit_btn.setEnabled(False)
        self.commit_btn.clicked.connect(self.commit)

        self.navigation_layout = QHBoxLayout()
        self.navigation_layout.addWidget(self.prev_img_btn)
        self.navigation_layout.addWidget(self.prev_patch_btn)
        self.navigation_layout.addWidget(self.next_patch_btn)
        self.navigation_layout.addWidget(self.next_img_btn)

        self.edit_layout = QHBoxLayout()
        self.edit_layout.addWidget(self.edit_labels_btn)
        self.edit_layout.addWidget(self.commit_btn)

        self.manager_layout = QVBoxLayout()
        self.manager_layout.addWidget(self.labels_table_tw)
        self.manager_layout.addLayout(self.navigation_layout)
        self.manager_layout.addLayout(self.edit_layout)

        self.setLayout(self.manager_layout)

        self._active_label: Union[None, LabelItem] = None
        self._active_label_group: Union[None, LabelGroup] = None
        self._active_layer_channel: Union[None, LayerChannel] = None
        self._active_layers_group: Union[None, LayersGroup] = None
        self._active_group: Union[None, ImageGroup] = None

        self._transaction = None
        self._active_edit_layer: Union[None, Layer] = None

        self._requires_commit = False

    def add_labels(self, layer_channel: LayerChannel,
                   labels: Iterable[LabelItem]):
        new_label_group = LabelGroup(layer_channel)
        new_label_group.addChildren(labels)

        self.labels_group_root.addChild(new_label_group)

        new_label_group.setExpanded(False)
        new_label_group.sortChildren(0, Qt.SortOrder.DescendingOrder)

    def navigate(self, delta_patch_index=0, delta_image_index=0):
        if self._requires_commit:
            self.commit()

        self._active_label.setSelected(False)

        self._active_label_group = self._active_label.parent()
        patch_index = self._active_label_group.indexOfChild(
            self._active_label
        )

        labels_group_index = self.labels_group_root.indexOfChild(
            self._active_label_group
        )

        if delta_patch_index:
            patch_index += delta_patch_index
            if patch_index >= self._active_label_group.childCount():
                patch_index = 0
                delta_image_index = 1

            elif patch_index < 0:
                delta_image_index = -1

        if delta_image_index:
            n_label_groups = self.labels_group_root.childCount()

            self._active_group = None
            self._active_layers_group = None
            self._active_layer_channel = None

            patch_index = 0 if delta_image_index > 0 else -1

            labels_group_index += delta_image_index
            labels_group_index = labels_group_index % n_label_groups

        self._active_label_group.setExpanded(False)
        self._active_label_group = self.labels_group_root.child(
            labels_group_index
        )
        self._active_label_group.setExpanded(True)

        patch_index = patch_index % self._active_label_group.childCount()

        self._active_label = self._active_label_group.child(patch_index)
        self._active_label.setSelected(True)

    def focus_region(self, label: Optional[QTreeWidgetItem] = None):
        if not label:
            label = self.labels_table_tw.selectedItems()

        if self._requires_commit:
            self.commit()

        if isinstance(label, list) and len(label):
            label = label[0]
        else:
            label = None

        if not isinstance(label, LabelItem):
            self.prev_img_btn.setEnabled(False)
            self.prev_patch_btn.setEnabled(False)
            self.next_patch_btn.setEnabled(False)
            self.next_img_btn.setEnabled(False)

            self.edit_labels_btn.setEnabled(False)

            return

        self._active_label = label

        self._active_label_group = self._active_label.parent()

        self._active_layer_channel = self._active_label_group.layer_channel
        self._active_layers_group = self._active_layer_channel.parent()
        self._active_group = self._active_layers_group.parent()

        current_center = [
            pos * ax_scl
            for pos, ax_scl in zip(self._active_label.center,
                                   self._active_layer_channel.layer.scale)
        ]

        self.viewer.dims.order = tuple(range(self.viewer.dims.ndim))
        self.viewer.camera.center = (
            *self.viewer.camera.center[:-self.viewer.dims.ndisplay],
            *current_center
        )
        self.viewer.camera.zoom = 4 / min(
            self._active_layer_channel.layer.scale
        )

        for layer in self.viewer.layers:
            layer.visible = False

        self._active_group.visible = True

        self.prev_img_btn.setEnabled(True)
        self.prev_patch_btn.setEnabled(True)
        self.next_patch_btn.setEnabled(True)
        self.next_img_btn.setEnabled(True)
        self.edit_labels_btn.setEnabled(True)

    def edit_labels(self):
        if (not self._active_layers_group or not self._active_label
           or not self._active_label):
            return

        input_filename = self._active_layers_group.source_data
        data_group = self._active_layers_group.data_group

        if isinstance(input_filename, (Path, str)):
            spec = {
                'driver': 'zarr',
                'kvstore': {
                    'driver': 'file',
                    'path': str(Path(input_filename) / data_group),
                },
            }

            ts_array = ts.open(spec).result()

            self._transaction = ts.Transaction()

            lazy_data = ts_array.with_transaction(self._transaction)
            lazy_data =\
                lazy_data[ts.d[:][self._active_label.position].translate_to[0]]

        elif isinstance(input_filename, MultiScaleData):
            lazy_data = np.array(input_filename[0][self._active_label.position])
        else:
            lazy_data = np.array(input_filename[self._active_label.position])

        self._active_edit_layer = self.viewer.add_labels(
            lazy_data,
            name="Labels edit",
            blending="translucent_no_depth",
            opacity=0.7,
            translate=[
                ax_roi.start * ax_scl
                for ax_roi, ax_scl in zip(
                    self._active_label.position,
                    self._active_layer_channel.layer.scale
                )
            ],
            scale=self._active_layer_channel.layer.scale
        )
        self.viewer.layers["Labels edit"].bounding_box.visible = True
        self._active_layer_channel.layer.visible = False

        self._requires_commit = True
        self.commit_btn.setEnabled(True)

    def commit(self):
        edit_data = None
        if self._active_edit_layer:
            edit_data = self._active_edit_layer.data

        segmentation_channel_layer = None
        segmentation_channel_data = None
        if self._active_layers_group:
            segmentation_channel = self._active_layers_group.child(0)
            segmentation_channel_layer = segmentation_channel.layer
            if isinstance(segmentation_channel.layer.data, MultiScaleData):
                segmentation_channel_data = segmentation_channel_layer.data[0]
            else:
                segmentation_channel_data = segmentation_channel_layer.data

        if isinstance(self._transaction, ts.Transaction):
            self._transaction.commit_async()
        elif (self._active_label.position is not None
              and segmentation_channel_data is not None):
            segmentation_channel_data[self._active_label.position] = edit_data

        if segmentation_channel_layer:
            segmentation_channel_layer.refresh()
            segmentation_channel_layer.visible = True

        if (self._active_edit_layer
           and self._active_edit_layer in self.viewer.layers):
            self.viewer.layers.remove(self._active_edit_layer)

        self._transaction = None
        self._active_edit_layer = None

        self._requires_commit = False
        self.commit_btn.setEnabled(False)


class AcquisitionFunction(QWidget):
    def __init__(self, viewer: napari.Viewer,
                 image_groups_manager: ImageGroupsManager,
                 labels_manager: LabelsManager,
                 model: Optional[CellposeModel] = None,
                 model_drop: Optional[CellposeModel] = None):
        super().__init__()

        self.viewer = viewer

        self.model = model
        self.model_drop = model_drop

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

        self.execute_selected_btn = QPushButton("Run on selected image groups")
        self.execute_selected_btn.clicked.connect(
            partial(self.compute_acquisition_layers, run_all=False,
                    model=self.model,
                    model_drop=self.model_drop)
        )

        self.execute_all_btn = QPushButton("Run on all image groups")
        self.execute_all_btn.clicked.connect(
            partial(self.compute_acquisition_layers, run_all=True,
                    model=self.model,
                    model_drop=self.model_drop)
        )

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

        self.fine_tune_btn = QPushButton("Fine tune model")
        self.fine_tune_btn.clicked.connect(
            partial(self.fine_tune)
        )

        self.execute_lyt = QHBoxLayout()
        self.execute_lyt.addWidget(self.execute_selected_btn)
        self.execute_lyt.addWidget(self.execute_all_btn)

        self.acquisition_lyt = QVBoxLayout()
        self.acquisition_lyt.addLayout(self.patch_size_lyt)
        self.acquisition_lyt.addLayout(self.max_samples_lyt)
        self.acquisition_lyt.addLayout(self.MC_repetitions_lyt)
        self.acquisition_lyt.addLayout(self.execute_lyt)
        self.acquisition_lyt.addLayout(self.image_lyt)
        self.acquisition_lyt.addLayout(self.patch_lyt)
        self.acquisition_lyt.addWidget(self.fine_tune_btn)
        self.setLayout(self.acquisition_lyt)

        self.image_groups_manager = image_groups_manager
        self.labels_manager = labels_manager

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
    def _compute_acquisition_fun(model_dropout, img, img_sp, MC_repetitions):
        probs = []
        for _ in range(MC_repetitions):
            probs.append(
                cellpose_probs(img[0].numpy(), model_dropout)
            )
        probs = np.stack(probs, axis=0)

        u_sp_lab = compute_acquisition(probs, img_sp[0, ..., 0].numpy())

        return u_sp_lab

    @staticmethod
    def _compute_segmentation(model, img, labels_offset=0):
        seg_out = cellpose_segment(img[0].numpy(), model)
        seg_out = np.where(seg_out, seg_out + labels_offset, 0)
        return seg_out

    def compute_acquisition(self, dataset_metadata, acquisition_fun,
                            segmentation_out,
                            sampled_pos=None,
                            MC_repetitions=30,
                            max_samples=1000,
                            patch_size=128,
                            model=None,
                            model_drop=None):
        dl = get_dataloader(dataset_metadata, patch_size=patch_size,
                            shuffle=True)
        segmentation_max = 0
        n_samples = 0
        img_sampling_positions = []

        self.patch_pb.setRange(0, max_samples)
        self.patch_pb.reset()
        for pos, img, img_sp in dl:
            pos_u_lab = (slice(pos[0, 0, 0].item(), pos[0, 0, 1].item()),
                         slice(pos[0, 1, 0].item(), pos[0, 1, 1].item()))
            sampled_pos_u_lab = (
                slice(pos[0, 0, 0].item() // patch_size,
                      pos[0, 0, 1].item() // patch_size),
                slice(pos[0, 1, 0].item() // patch_size,
                      pos[0, 1, 1].item() // patch_size)
            )

            if model_drop:
                u_sp_lab = self._compute_acquisition_fun(
                    model_drop,
                    img,
                    img_sp,
                    MC_repetitions
                )
                acquisition_fun[pos_u_lab] = u_sp_lab
                acquisition_val = u_sp_lab.max()
            else:
                acquisition_val = 0

            if model:
                seg_out = self._compute_segmentation(
                    model,
                    img,
                    segmentation_max
                )
                segmentation_out[pos_u_lab] = seg_out
                segmentation_max = max(segmentation_max, seg_out.max())

            if sampled_pos is not None:
                sampled_pos[sampled_pos_u_lab] = True

            img_sampling_positions.append(
                LabelItem(acquisition_val, position=pos_u_lab)
            )

            n_samples += 1
            self.patch_pb.setValue(n_samples)
            if n_samples >= max_samples:
                break

        self.patch_pb.setValue(max_samples)
        return img_sampling_positions

    def compute_acquisition_layers(self, run_all: bool = False,
                                   model: Optional[CellposeModel] = None,
                                   model_drop: Optional[CellposeModel] = None
                                   ):
        if run_all:
            for idx in range(self.image_groups_manager.groups_root.childCount()
                             ):
                child = self.image_groups_manager.groups_root.child(idx)
                child.setSelected(isinstance(child, ImageGroup))

        image_groups: List[ImageGroup] = list(filter(
            lambda item:
            isinstance(item, ImageGroup),
            self.image_groups_manager.image_groups_tw.selectedItems()
        ))

        if not image_groups:
            return

        MC_repetitions = self.MC_repetitions_spn.value()
        max_samples = self.max_samples_spn.value()
        patch_size = self.patch_size_spn.value()

        self._update_roi_from_position()

        self.image_pb.setRange(0, len(image_groups))
        self.image_pb.reset()

        for n, image_group in enumerate(image_groups):
            image_group.setSelected(True)
            group_name = image_group.group_name
            if image_group.group_dir:
                output_filename = image_group.group_dir / (group_name
                                                           + ".zarr")
            else:
                output_filename = None

            input_layers_group_idx = image_group.input_layers_group
            if input_layers_group_idx is None:
                continue

            input_layers_group = image_group.child(input_layers_group_idx)
            sampled_positions_layers_group = image_group.getLayersGroup(
                "sampled-positions"
            )

            displayed_source_axes = input_layers_group.source_axes
            displayed_shape = input_layers_group.shape
            displayed_scale = input_layers_group.scale

            acquisition_fun_shape, acquisition_fun_scale = list(zip(*[
                (ax_s, ax_scl)
                for ax, ax_s, ax_scl in zip(displayed_source_axes,
                                            displayed_shape,
                                            displayed_scale)
                if ax in "ZYX" and ax_s > 1
            ]))

            if model_drop:
                acquisition_root = save_zarr(
                    output_filename,
                    data=None,
                    shape=acquisition_fun_shape,
                    chunk_size=True,
                    name="acquisition_fun",
                    dtype=np.float32,
                    is_label=True,
                    is_multiscale=True
                )

                acquisition_fun_grp =\
                    acquisition_root["labels/acquisition_fun/0"]
            else:
                acquisition_fun_grp = None

            if model:
                segmentation_root = save_zarr(
                    output_filename,
                    data=None,
                    shape=acquisition_fun_shape,
                    chunk_size=True,
                    name="segmentation",
                    dtype=np.int32,
                    is_label=True,
                    is_multiscale=True
                )

                segmentation_grp = segmentation_root["labels/segmentation/0"]
            else:
                segmentation_grp = None

            if sampled_positions_layers_group:
                sampled_pos_grp = None
                sampling_mask_layers_group = sampled_positions_layers_group
            else:
                if image_group.sampling_mask_layers_group is not None:
                    sampling_mask_layers_group = image_group.child(
                        image_group.sampling_mask_layers_group
                    )
                else:
                    sampling_mask_layers_group = None

                (sampled_pos_shape,
                 sampled_pos_scale,
                 sampled_pos_translate) = list(zip(*[
                     (ax_s // patch_size,
                      ax_scl * patch_size,
                      ((ax_scl * (patch_size - 1) / 2)
                       if ax_s // patch_size >= 1 else 0))
                     for ax, ax_s, ax_scl in zip(displayed_source_axes,
                                                 displayed_shape,
                                                 displayed_scale)
                     if ax in "ZYX" and ax_s > 1]))

                sampled_pos_root = save_zarr(
                    output_filename,
                    data=None,
                    shape=sampled_pos_shape,
                    chunk_size=True,
                    name="sampled_positions",
                    dtype=bool,
                    is_label=True,
                    is_multiscale=True
                )

                sampled_pos_grp = sampled_pos_root["labels"
                                                   "/sampled_positions/0"]

            dataset_metadata = {}

            for layers_group, layer_type in [
               (input_layers_group, "images"),
               (sampling_mask_layers_group, "masks")
               ]:
                if layers_group is None:
                    continue

                dataset_metadata[layer_type] = layers_group.metadata
                dataset_metadata[layer_type]["roi"] = [
                    tuple(
                        self._roi.get(ax, slice(None))
                        for ax in layers_group.source_axes
                    )
                ]

                spatial_axes = "".join([
                    ax for ax in layers_group.source_axes
                    if ax in displayed_source_axes[-self.viewer.dims.ndisplay:]
                ])

                if "images" in layer_type:
                    if "C" in displayed_source_axes:
                        spatial_axes += "C"

                if isinstance(dataset_metadata[layer_type]["filenames"],
                              MultiScaleData):
                    dataset_metadata[layer_type]["filenames"] =\
                        dataset_metadata[layer_type]["filenames"][0]

                dataset_metadata[layer_type]["axes"] = spatial_axes
                dataset_metadata[layer_type]["modality"] = layer_type

            # Compute acquisition function of the current image
            img_sampling_positions = self.compute_acquisition(
                dataset_metadata,
                acquisition_fun=acquisition_fun_grp,
                segmentation_out=segmentation_grp,
                sampled_pos=sampled_pos_grp,
                MC_repetitions=MC_repetitions,
                max_samples=max_samples,
                patch_size=patch_size,
                model=model,
                model_drop=model_drop
            )
            self.image_pb.setValue(n + 1)

            if not img_sampling_positions:
                continue

            if model_drop:
                if output_filename:
                    acquisition_root = output_filename

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

                new_acquisition_layer = viewer.add_image(
                    acquisition_fun_ms,
                    name=group_name + " acquisition function",
                    multiscale=True,
                    opacity=0.8,
                    scale=acquisition_fun_scale,
                    blending="translucent_no_depth",
                    colormap="magma",
                    contrast_limits=(
                        0,
                        max(img_sampling_positions).acquisition_val
                    ),
                )

                if isinstance(new_acquisition_layer, list):
                    new_acquisition_layer = new_acquisition_layer[0]

                acquisition_layers_group = image_group.getLayersGroup(
                    "acquisition"
                )

                if acquisition_layers_group is None:
                    acquisition_layers_group = image_group.add_layers_group(
                        "acquisition",
                        source_axes="YX",
                        use_as_input_image=False,
                        use_as_sampling_mask=False
                    )

                acquisition_channel = acquisition_layers_group.add_layer(
                    new_acquisition_layer
                )

                if output_filename:
                    acquisition_channel.source_data = str(output_filename)
                    acquisition_channel.data_group = "labels/acquisition_fun"

            if model:
                if output_filename:
                    segmentation_root = output_filename

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

                segmentation_ms_kwargs = dict(
                    name=group_name + " segmentation",
                    multiscale=True,
                    opacity=0.8,
                    scale=acquisition_fun_scale,
                    blending="translucent_no_depth",
                )

                new_segmentation_layer = viewer.add_labels(
                    segmentation_ms,
                    **segmentation_ms_kwargs
                )

                if isinstance(new_segmentation_layer, list):
                    new_segmentation_layer = new_segmentation_layer[0]

                segmentation_layers_group = image_group.getLayersGroup(
                    "segmentation"
                )

                if segmentation_layers_group is None:
                    segmentation_layers_group = image_group.add_layers_group(
                        "segmentation",
                        source_axes="YX",
                        use_as_input_image=False,
                        use_as_sampling_mask=False
                    )

                segmentation_channel = segmentation_layers_group.add_layer(
                    new_segmentation_layer
                )

                if model_drop:
                    self.labels_manager.add_labels(
                        segmentation_channel,
                        img_sampling_positions
                    )

                if output_filename:
                    segmentation_channel.source_data = str(output_filename)
                    segmentation_channel.data_group = "labels/segmentation"

            if sampled_pos_grp:
                if output_filename:
                    sampled_pos_root = output_filename

                # Downsample the segmentation output
                sampled_pos_ms = downsample_image(
                    sampled_pos_root,
                    source_axes="YX",
                    data_group="labels/sampled_positions/0",
                    scale=4,
                    num_scales=0,
                    reference_source_axes=displayed_source_axes,
                    reference_scale=displayed_scale,
                )

                sampled_pos_kwargs = dict(
                    name=group_name + " sampled positions",
                    multiscale=True,
                    opacity=0.8,
                    translate=sampled_pos_translate,
                    scale=sampled_pos_scale,
                    blending="translucent_no_depth",
                )

                new_sampled_pos_layer = viewer.add_labels(
                    np.array(sampled_pos_ms),
                    **sampled_pos_kwargs
                )

                if isinstance(new_sampled_pos_layer, list):
                    new_sampled_pos_layer = new_sampled_pos_layer[0]

                sampled_pos_layers_group = image_group.getLayersGroup(
                    "sampled-positions"
                )

                if sampled_pos_layers_group is None:
                    sampled_pos_layers_group = image_group.add_layers_group(
                        "sampled-positions",
                        source_axes="YX",
                        use_as_input_image=False,
                        use_as_sampling_mask=True
                    )

                sampled_pos_channel = sampled_pos_layers_group.add_layer(
                    new_sampled_pos_layer
                )

                if output_filename:
                    sampled_pos_channel.source_data = \
                        napari.layers._source.Source(
                            path=str(output_filename
                                     / "labels/sampled_positions")
                        )

    def fine_tune(self):
        image_groups: List[ImageGroup] = list(filter(
            lambda item:
            isinstance(item, ImageGroup),
            map(lambda idx:
                self.image_groups_manager.groups_root.child(idx),
                range(self.image_groups_manager.groups_root.childCount()))
        ))

        if not image_groups:
            return

        patch_size = self.patch_size_spn.value()

        model = cellpose_model_init()

        dataset_metadata_list = []

        for image_group in image_groups:
            image_group.setSelected(True)

            input_layers_group_idx = image_group.input_layers_group
            segmentation_layers_group = image_group.getLayersGroup(
                layers_group_name="segmentation"
            )
            sampled_pos_layers_group = image_group.getLayersGroup(
                layers_group_name="sampled-positions"
            )

            if (input_layers_group_idx is None
               or segmentation_layers_group is None
               or sampled_pos_layers_group is None):
                continue

            input_layers_group = image_group.child(input_layers_group_idx)

            displayed_source_axes = input_layers_group.source_axes

            dataset_metadata = {}

            for layers_group, layer_type in [
               (input_layers_group, "images"),
               (segmentation_layers_group, "labels"),
               (sampled_pos_layers_group, "masks")
               ]:
                dataset_metadata[layer_type] = layers_group.metadata
                dataset_metadata[layer_type]["roi"] = [
                    tuple(
                        self._roi.get(ax, slice(None))
                        for ax in layers_group.source_axes
                    )
                ]

                spatial_axes = "".join([
                    ax for ax in layers_group.source_axes
                    if ax in displayed_source_axes[-self.viewer.dims.ndisplay:]
                ])

                if "images" in layer_type:
                    if "C" in displayed_source_axes:
                        spatial_axes += "C"

                if isinstance(dataset_metadata[layer_type]["filenames"],
                              MultiScaleData):
                    dataset_metadata[layer_type]["filenames"] =\
                        dataset_metadata[layer_type]["filenames"][0]

                if isinstance(dataset_metadata[layer_type]["filenames"],
                              da.core.Array):
                    dataset_metadata[layer_type]["filenames"] =\
                        dataset_metadata[layer_type]["filenames"].compute()

                dataset_metadata[layer_type]["axes"] = spatial_axes
                dataset_metadata[layer_type]["modality"] = layer_type

            dataset_metadata_list.append(dataset_metadata)

        model = train_cellpose(model, dataset_metadata_list,
                               patch_size=patch_size,
                               spatial_axes=spatial_axes,
                               n_epochs=10)

        self.compute_acquisition_layers(run_all=True, model=model)


class MaskGenerator(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.viewer = viewer

        self._active_image_group: Union[None, ImageGroup] = None
        self._active_layers_group: Union[None, LayersGroup] = None

        self.patch_size_lbl = QLabel("Patch size")

        self.patch_size_le = QLineEdit("128")
        self.patch_size_le.setValidator(QIntValidator(0, 2**16))
        self.patch_size_spn = QSpinBox(minimum=0, maximum=16, singleStep=1)
        self.patch_size_spn.setValue(7)
        self.patch_size_spn.lineEdit().hide()
        self.patch_size_spn.setEnabled(False)
        self.patch_size_spn.valueChanged.connect(self.set_patch_size)

        self.generate_mask_btn = QPushButton("Create mask")
        self.generate_mask_btn.setToolTip("Create a napari Label layer with a "
                                          "blank mask at the scale selected "
                                          "with `Patch size`")
        self.generate_mask_btn.setEnabled(False)
        self.generate_mask_btn.clicked.connect(self.generate_mask_layer)

        self.create_mask_lyt = QHBoxLayout()
        self.create_mask_lyt.addWidget(self.patch_size_lbl)
        self.create_mask_lyt.addWidget(self.patch_size_le)
        self.create_mask_lyt.addWidget(self.patch_size_spn)
        self.create_mask_lyt.addWidget(self.generate_mask_btn)

        self.setLayout(self.create_mask_lyt)

    @property
    def active_layers_group(self):
        return self._active_layers_group

    @active_layers_group.setter
    def active_layers_group(self, active_layers_group: LayersGroup):
        self._active_layers_group = active_layers_group

        self.generate_mask_btn.setEnabled(
            self._active_image_group is not None
            and self._active_layers_group is not None
        )
        self.patch_size_le.setEnabled(
            self._active_image_group is not None
            and self._active_layers_group is not None
        )
        self.patch_size_spn.setEnabled(
            self._active_image_group is not None
            and self._active_layers_group is not None
        )

    @property
    def active_image_group(self):
        return self._active_image_group

    @active_image_group.setter
    def active_image_group(self, active_image_group: ImageGroup):
        self._active_image_group = active_image_group

        self.generate_mask_btn.setEnabled(
            self._active_image_group is not None
            and self._active_layers_group is not None
        )
        self.patch_size_le.setEnabled(
            self._active_image_group is not None
            and self._active_layers_group is not None
        )
        self.patch_size_spn.setEnabled(
            self._active_image_group is not None
            and self._active_layers_group is not None
        )

    def set_patch_size(self, new_value: int):
        self.patch_size_le.setText(str(2 ** new_value))

    def generate_mask_layer(self):
        if (not self._active_image_group
           or not self._active_layers_group
           or not self._active_layers_group.childCount()
           or not self._active_layers_group.source_axes):
            return

        # patch_size = self.patch_size_spn.value()
        patch_size = int(self.patch_size_le.text())

        source_axes = [
            ax
            for ax in self._active_layers_group.source_axes
            if ax != "C"
        ]

        reference_layer = self._active_layers_group.child(0).layer
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
            (ax_scl * patch_size) if ax_s // patch_size >= 1 else ax_scl
            for ax, ax_s, ax_scl in zip(source_axes, im_shape, im_scale)
            if ax in mask_axes
        )

        mask_translate = tuple(
            ax_trans
            + ((ax_scl * (patch_size - 1) / 2)
               if ax_s // patch_size >= 1 else 0)
            for ax, ax_s, ax_scl, ax_trans in zip(source_axes, im_shape,
                                                  im_scale,
                                                  im_trans)
            if ax in mask_axes
        )

        new_mask_layer = self.viewer.add_labels(
            data=np.zeros(mask_shape, dtype=np.uint8),
            scale=mask_scale,
            translate=mask_translate,
            name=self._active_layers_group.layers_group_name + " sampling mask",
        )

        masks_layers_group = self._active_image_group.getLayersGroup("masks")
        if masks_layers_group is None:
            masks_layers_group = self._active_image_group.add_layers_group(
                "masks",
                source_axes=mask_axes,
                use_as_sampling_mask=True
            )

        masks_layers_group_index = self._active_image_group.indexOfChild(
            masks_layers_group
        )

        self._active_image_group.sampling_mask_layers_group =\
            masks_layers_group_index

        masks_layers_group.add_layer(new_mask_layer)


if __name__ == "__main__":
    viewer = napari.Viewer()

    image_groups_manager = ImageGroupsManager(viewer)
    viewer.window.add_dock_widget(image_groups_manager, area='right')

    labels_manager = LabelsManager(viewer)

    model_drop = cellpose_model_init(use_dropout=True)
    model = cellpose_model_init(use_dropout=False)

    acquisition_function = AcquisitionFunction(viewer, image_groups_manager,
                                               labels_manager=labels_manager,
                                               model=model,
                                               model_drop=model_drop)
    viewer.window.add_dock_widget(acquisition_function, area='right')

    viewer.window.add_dock_widget(labels_manager, area='right')

    napari.run()
