from typing import List, Iterable, Union, Optional
from itertools import repeat

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
                            QLineEdit,
                            QComboBox,
                            QLabel,
                            QFileDialog,
                            QSpinBox,
                            QProgressBar,
                            QTreeWidget,
                            QTreeWidgetItem)

from functools import partial
from pathlib import Path
import torch
import napari
from napari.layers import Image, Layer
from napari.layers._multiscale_data import MultiScaleData
import tensorstore as ts
import numpy as np
import zarr

from ome_zarr.writer import write_multiscales_metadata, write_label_metadata

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
    def __init__(self, layer: Layer, channel:int = 0,
                 source_axes: str = "TZYX"):
        self.layer = layer
        self._channel = channel
        self._source_axes = source_axes

        super().__init__([layer.name, "", str(self._channel), source_axes])

        layer.events.name.connect(self._update_name)

        self._source_data = None
        self._data_group = None

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


class LayersGroup(QTreeWidgetItem):
    def __init__(self, layer_type: str,
                 layer_group_name: Optional[str] = None,
                 source_axes: Union[str, None] = None,
                 add_to_output: bool = False):

        self._layer_group_name = layer_group_name
        self._layer_type = layer_type

        self._source_axes_no_channels = None
        self._source_axes = None
        self._source_data = None
        self._data_group = None

        self._updated = True
        self._add_to_output = add_to_output

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
            "add_to_output": self._add_to_output
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
    def add_to_output(self):
        return self._add_to_output

    @add_to_output.setter
    def add_to_output(self, add_to_output: bool):
        self._add_to_output = add_to_output

    def add_layer(self, layer: Layer, channel: Union[int, None] = None,
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

        new_layer_channel.setExpanded(True)

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
        child = super(LayersGroup, self).takeChild(index)
        self.remove_layer(child)
        return child

    def removeChild(self, child: QTreeWidgetItem):
        self.remove_layer(child)
        super(LayersGroup, self).removeChild(child)

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


class SamplingMaskGroup(LayersGroup):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_mask(self, name="sampling_mask", output_dir=Path(".")):
        output_filename = output_dir / (self.group_name + ".zarr")

        mask_data_group = "labels/" + name
        mask_source_data = self.source_data

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

        save_zarr(output_filename, data=mask_source_data,
                  shape=mask_source_data.shape,
                  chunk_size=True,
                  group_name=mask_data_group,
                  dtype=bool,
                  metadata=mask_metadata)

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

    @property
    def metadata(self):
        if self._group_metadata is None or self._updated_groups:
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

    def add_layer(self, layer: Layer, layer_type: Union[str, None] = None,
                  channel: Union[int, None] = None,
                  source_axes: Union[str, None] = None,
                  add_to_output: bool = False):
        if not self._group_name:
            self.group_name = get_basename(layer.name)

        if layer_type is None:
            layer_type = "images" if isinstance(layer, Image) else "masks"

        layers_group = list(filter(
            lambda curr_layers_group:
            curr_layers_group.layer_type == layer_type,
            map(lambda idx: self.child(idx), range(self.childCount()))
        ))

        if layers_group:
            layers_group = layers_group[0]

        if not layers_group:
            if "mask" in layer_type:
                layers_group = SamplingMaskGroup(
                    layer_type,
                    source_axes=source_axes,
                    add_to_output=add_to_output
                )
            else:
                layers_group = LayersGroup(
                    layer_type,
                    source_axes=source_axes,
                    add_to_output=add_to_output
                )

            self.addChild(layers_group)

        new_layer_channel = layers_group.add_layer(
            layer,
            channel=channel,
            source_axes=source_axes
        )

        self._updated_groups.add(layer_type)

        self._updated = True

        layers_group.setExpanded(True)

        return new_layer_channel

    def move_layer(self, layer_type: Union[str, None],
                   layers_group: LayersGroup,
                   layer_channel: Union[LayerChannel, None] = None):
        if layer_channel:
            layer_channel_list = [
                layers_group.takeChild(
                    layers_group.indexOfChild(layer_channel)
                )
            ]

        else:
            layer_channel_list = layers_group.takeChildren()

        if layer_type:
            for curr_layer_channel in layer_channel_list:
                new_layers_group = self.add_layer(
                    layer_type,
                    curr_layer_channel.layer,
                    channel=None,
                    source_axes=curr_layer_channel.source_axes
                )
        else:
            new_layers_group = None

        if not layers_group.childCount():
            self.removeChild(layers_group)

        return new_layers_group

    def update_group_metadata(self):
        layers_groups = list(filter(
            lambda curr_layers_group:
            curr_layers_group.layer_type in self._updated_groups,
            map(lambda idx: self.child(idx), range(self.childCount()))
        ))

        for layers_group in layers_groups:
            self._group_metadata[layers_group.layer_type] =\
                 layers_group.metadata

        self._updated_groups.clear()


class ImageGroupRoot(QTreeWidgetItem):
    def __init__(self):
        super().__init__(["Image groups"])
        self.managed_layers = {}
        self.group_names = set()

    def add_managed_layer(self, image_group: ImageGroup,
                          layer: Layer,
                          **kwargs):
        if layer not in self.managed_layers:
            self.managed_layers[layer] = []

        if not image_group.group_name:
            layer_name = layer.name

            if layer_name in self.group_names:
                n_existing = sum(map(
                    lambda exisitng_group_name:
                    layer_name in exisitng_group_name,
                    self.group_names
                ))

                layer_name = layer_name + " (%i)" % n_existing

            self.group_names.add(layer_name)
            image_group.group_name = layer_name

        layer_channel = image_group.add_layer(layer, **kwargs)

        self.managed_layers[layer].append(layer_channel)

        return layer_channel

    def remove_managed_layer(self, event):
        removed_layer = event.value

        layer_channel_list = self.managed_layers.get(removed_layer, [])
        for layer_channel in layer_channel_list:
            layers_group = layer_channel.parent()
            layers_group.removeChild(layer_channel)

        if layer_channel_list:
            self.managed_layers.pop(removed_layer)


class ImageGroupEditor(QWidget):

    updated = Signal()

    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.viewer = viewer

        self._active_image_group = None
        self._active_layers_group = None
        self._active_layer_channel = None

        self.group_name_lbl = QLabel("Group name:")
        self.group_name_le = QLineEdit("None selected")
        self.group_name_le.setEnabled(False)

        self.layer_name_lbl = QLabel("Layer name:")
        self.display_name_lbl = QLabel("None selected")

        self.layer_type_lbl = QLabel("Layer type:")
        self.layer_type_cmb = QComboBox()
        self.layer_type_cmb.setEditable(True)
        self.layer_type_cmb.setEnabled(False)

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

        self.layer_type_cmb.lineEdit().returnPressed.connect(
            self._update_metadata
        )
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
        # self.edit_2_lyt.addWidget(self.layer_type_le)
        self.edit_2_lyt.addWidget(self.layer_type_cmb)

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
        new_layer_type = self.layer_type_cmb.currentText()
        new_source_axes = self.edit_axes_le.text().upper()
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

        if self._active_layer_channel:
            if self._active_layer_channel.source_axes != new_source_axes:
                self._active_layer_channel.source_axes = new_source_axes

            prev_channel = self._active_layer_channel.channel
            if prev_channel != new_channel:
                self._active_layers_group.move_channel(prev_channel,
                                                       new_channel)

        if self._active_layers_group:
            if self._active_layers_group.source_axes != new_source_axes:
                self._active_layers_group.source_axes = new_source_axes

            if self._active_layers_group.layer_type != new_layer_type:
                self._active_layers_group = self._active_image_group.move_layer(
                    new_layer_type,
                    self._active_layers_group,
                    layer_channel=self._active_layer_channel
                )

                self._active_layer_channel = None

        if self._active_image_group:
            if self._active_image_group.group_name != new_group_name:
                self._active_image_group.group_name = new_group_name

            if self._active_image_group.group_dir != new_output_dir:
                self._active_image_group.group_dir = new_output_dir

        self.updated.emit()

    def _clear_image_group(self):
        # Clear image group
        self.layer_type_cmb.clear()
        self.output_dir_le.setText("None selected")
        self.group_name_le.setText("None selected")

        self.output_dir_btn.setEnabled(False)
        self.output_dir_le.setEnabled(False)
        self.group_name_le.setEnabled(False)

    def _clear_layers_group(self):
        # Clear layer channels group
        # self.layer_type_le.setText("None selected")
        # self.layer_type_le.setEnabled(False)

        self.edit_axes_le.setText("None selected")
        self.remove_layer_group_btn.setEnabled(False)
        self.layer_type_cmb.setEnabled(False)
        self.edit_axes_le.setEnabled(False)

    def _clear_layer_channel(self):
        # Clear layer channel
        self.display_name_lbl.setText("None selected")

        self.edit_channel_spn.setValue(0)
        self.edit_channel_spn.setMaximum(0)
        self.edit_channel_spn.setEnabled(False)
        self.remove_layer_btn.setEnabled(False)

    def _update_image_group(self):
        self._clear_image_group()

        if self._active_image_group:
            self.output_dir_le.setText(str(self._active_image_group.group_dir))
            self.group_name_le.setText(self._active_image_group.group_name)

            self.layer_type_cmb.clear()
            for idx in range(self._active_image_group.childCount()):
                layers_group = self._active_image_group.child(idx)
                self.layer_type_cmb.addItem(layers_group.layer_type)

            self.output_dir_btn.setEnabled(True)
            self.output_dir_le.setEnabled(True)
            self.group_name_le.setEnabled(True)

        self._update_layers_group()

    def _update_layers_group(self):
        self._clear_layers_group()

        if self._active_layers_group:
            self.layer_type_cmb.lineEdit().setText(
                self._active_layers_group.layer_type
            )
            self.edit_axes_le.setText(
                self._active_layers_group.source_axes
            )
            self.remove_layer_group_btn.setEnabled(True)
            self.layer_type_cmb.setEnabled(True)
            self.edit_axes_le.setEnabled(True)

        self._update_layer_channel()

    def _update_layer_channel(self):
        self._clear_layer_channel()

        if self._active_layer_channel:
            self.display_name_lbl.setText(self._active_layer_channel.layer.name)
            self.edit_axes_le.setText(self._active_layer_channel.source_axes)
            self.remove_layer_btn.setEnabled(True)
            self.edit_axes_le.setEnabled(True)

            self.edit_channel_spn.setMaximum(
                self.active_layers_group.childCount() - 1
            )
            self.edit_channel_spn.setValue(self._active_layer_channel.channel)
            self.edit_channel_spn.setEnabled(True)

    def remove_layer_group(self):
        if not self._active_layers_group:
            return

        self._active_image_group.removeChild(self._active_layers_group)

        self._update_image_group()

    def remove_layer(self):
        if not (self._active_layers_group and self._active_layer_channel):
            return

        self._active_layers_group.removeChild(self.active_layer_channel)

        self._active_layer_channel = None
        self._update_layers_group()

    @property
    def active_image_group(self):
        return self._active_image_group

    @active_image_group.setter
    def active_image_group(self,
                           active_image_group: Union[ImageGroup, None] = None
                           ):
        self._active_image_group = active_image_group
        self._active_layers_group = None
        self._active_layer_channel = None

        self._update_image_group()

    @property
    def active_layers_group(self):
        return self._active_layers_group

    @active_layers_group.setter
    def active_layers_group(self,
                                    active_layers_group: Union[
                                        LayersGroup,
                                        None] = None
                                    ):
        self._active_layers_group = active_layers_group
        self._active_layer_channel = None

        self._update_layers_group()

    @property
    def active_layer_channel(self):
        return self._active_layer_channel

    @active_layer_channel.setter
    def active_layer_channel(self, active_layer_channel: Union[LayerChannel, None] = None):
        self._active_layer_channel = active_layer_channel
        self._update_layer_channel()


class ImageGroupsManager(QWidget):

    layer_removed = Signal()
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
            axis_labels = ("".join(map(str, range(4, ndims)))
                           + default_axis_labels)

        self.viewer.dims.axis_labels = list(axis_labels)

        self.active_image_group = None
        self.active_layers_group = None
        self.active_layer_channel = None

        self.image_groups_tw = QTreeWidget()

        self.image_groups_tw.setColumnCount(5)
        self.image_groups_tw.setHeaderLabels(["Group name", "Layer type",
                                              "Channels",
                                              "Axes order",
                                              "Output directory"])

        self.image_groups_tw.itemSelectionChanged.connect(
            self._get_active_item
        )
        self.root_group = ImageGroupRoot()
        viewer.layers.events.removed.connect(
            self.root_group.remove_managed_layer
        )
        self.image_groups_tw.addTopLevelItem(self.root_group)
        self.root_group.setExpanded(True)

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

        self.group_lyt = QVBoxLayout()
        self.group_lyt.addLayout(self.buttons_lyt)
        self.group_lyt.addWidget(self.layers_editor)
        self.group_lyt.addWidget(self.mask_creator)
        self.group_lyt.addWidget(self.image_groups_tw)

        self.setLayout(self.group_lyt)

    def _get_active_item(self):
        item = self.image_groups_tw.selectedItems()
        if isinstance(item, list):
            if not len(item):
                return

            item = item[0]

        self.active_layer_channel = None
        self.active_layers_group = None
        self.active_image_group = None

        if isinstance(item, LayerChannel):
            self.active_layer_channel = item

        elif isinstance(item, LayersGroup):
            self.active_layers_group = item

        elif isinstance(item, ImageGroup) and item != self.root_group:
            self.active_image_group = item

        elif isinstance(item, LabelItem):
            self.active_layer_channel = item.parent().parent()

        elif isinstance(item, LabelGroup):
            self.active_layer_channel = item.parent()

        if self.active_layer_channel:
            self.active_layers_group = self.active_layer_channel.parent()

        if self.active_layers_group:
            self.active_image_group = self.active_layers_group.parent()

        self.mask_creator.active_image_group = self.active_image_group
        self.mask_creator.active_layers_group = self.active_layers_group

        self.layers_editor.active_image_group = self.active_image_group
        self.layers_editor.active_layers_group = self.active_layers_group
        self.layers_editor.active_layer_channel = self.active_layer_channel

        self.remove_group_btn.setEnabled(self.active_image_group is not None)
        self.add_group_btn.setEnabled(self.active_image_group is not None)

        self.layer_selected.emit(item)

    def update_group(self, new_layers: Iterable[Layer],
                     active_image_group: ImageGroup,
                     layer_types: Union[str, Iterable[str], None] = None,
                     active_source_axes: Union[str, None] = None,
                     add_to_output_list: Union[bool, Iterable[bool]] = True
                     ):
        self._updated = True

        if active_source_axes:
            if "C" in active_source_axes:
                active_source_axes = list(active_source_axes)
                active_source_axes.remove("C")
                active_source_axes = "".join(active_source_axes)

        if not isinstance(layer_types, list):
            layer_types = repeat(layer_types)

        if not isinstance(add_to_output_list, list):
            add_to_output_list = repeat(add_to_output_list)

        for layer, layer_type, add_to_output in zip(new_layers, layer_types,
                                                    add_to_output_list):
            self.root_group.add_managed_layer(
                active_image_group,
                layer,
                layer_type=layer_type,
                channel=None,
                source_axes=active_source_axes,
                add_to_output=add_to_output
            )

        active_image_group.setExpanded(True)

    def update_group_from_selected(self):
        selected_layers = sorted(
            map(lambda layer: (layer.name, layer),
                self.viewer.layers.selection)
        )

        selected_layers = list(zip(*selected_layers))[1]

        if not selected_layers or not self.active_image_group:
            return

        active_source_axes = "".join(self.viewer.dims.axis_labels).upper()
        if self.active_layers_group:
            active_source_axes = self.active_layers_group.source_axes

        self.update_group(new_layers=selected_layers,
                          active_image_group=self.active_image_group,
                          active_source_axes=active_source_axes)

    def create_group(self):
        self.active_image_group = ImageGroup()
        self.root_group.addChild(self.active_image_group)

        self.update_group_from_selected()

    def remove_group(self):
        self.root_group.removeChild(self.active_image_group)

        self.active_layer_channel = None
        self.active_layers_group = None
        self.active_image_group = None

        self._updated = True

        self.add_group_btn.setEnabled(False)
        self.remove_group_btn.setEnabled(False)


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
    def __init__(self):
        super().__init__(["Acquisition value",
                          "Sampling center",
                          "Sampling top-left",
                          "Sampling bottom-right"])


class LabelsManager(QWidget):
    def __init__(self, viewer: napari.Viewer,
                 image_groups_manager: ImageGroupsManager):
        super().__init__()

        self.viewer = viewer

        self.image_groups_manager = image_groups_manager

        self.image_groups_manager.layer_selected.connect(self.focus_region)

        self.prev_img_btn = QPushButton('<<')
        self.prev_img_btn.clicked.connect(partial(
            self.navigate, delta_image_index=-1
        ))

        self.prev_patch_btn = QPushButton('<')
        self.prev_patch_btn.clicked.connect(partial(
            self.navigate, delta_patch_index=-1
        ))

        self.next_patch_btn = QPushButton('>')
        self.next_patch_btn.clicked.connect(partial(
            self.navigate, delta_patch_index=1
        ))

        self.next_img_btn = QPushButton('>>')
        self.next_img_btn.clicked.connect(partial(
            self.navigate, delta_image_index=1
        ))

        self.fix_labels_btn = QPushButton("Fix current labels")
        self.fix_labels_btn.clicked.connect(self.fix_labels)

        self.commit_btn = QPushButton("Commit changes")
        self.commit_btn.setEnabled(False)
        self.commit_btn.clicked.connect(self.commit)

        self.nav_btn_layout = QHBoxLayout()
        self.nav_btn_layout.addWidget(self.prev_img_btn)
        self.nav_btn_layout.addWidget(self.prev_patch_btn)
        self.nav_btn_layout.addWidget(self.next_patch_btn)
        self.nav_btn_layout.addWidget(self.next_img_btn)

        self.fix_btn_layout = QVBoxLayout()
        self.fix_btn_layout.addWidget(self.fix_labels_btn)
        self.fix_btn_layout.addWidget(self.commit_btn)
        self.fix_btn_layout.addLayout(self.nav_btn_layout)

        self.setLayout(self.fix_btn_layout)

        self.active_label = None
        self.active_layers_group = None
        self.active_group = None
        self._transaction = None

        self._commited = True

    def navigate(self, delta_patch_index=0, delta_image_index=0):
        if not self._commited:
            self.commit()

        if (not self.active_group or not self.active_layers_group
           or not self.active_label):
            return

        labels_group = self.active_label.parent()
        patch_index = labels_group.indexOfChild(
            self.active_label
        )

        if delta_patch_index:
            patch_index += delta_patch_index
            if patch_index >= labels_group.childCount():
                patch_index = 0
                delta_image_index = 1

            elif patch_index < 0:
                delta_image_index = -1

        if delta_image_index:
            n_image_groups = self.image_groups_manager.root_group.childCount()

            image_index = self.image_groups_manager.root_group.indexOfChild(
                self.active_group
            )

            self.active_group = None
            self.active_layers_group = None

            patch_index = 0 if delta_image_index > 0 else -1

            while True:
                image_index += delta_image_index

                if image_index >= n_image_groups:
                    image_index = 0
                elif image_index < 0:
                    image_index = n_image_groups - 1

                image_group = self.image_groups_manager.root_group.child(
                    image_index
                )

                for idx in range(image_group.childCount()):
                    layer_group = image_group.child(idx)

                    for idx_grp in range(layer_group.childCount()):
                        layer_channel = layer_group.child(idx_grp)

                        for idx_ch in range(layer_channel.childCount()):
                            labels_group = layer_channel.child(idx_ch)

                            if isinstance(labels_group, LabelGroup):
                                self.active_group = image_group
                                self.active_layers_group = layer_group
                                break

                        else:
                            continue

                        break

                    else:
                        continue

                    break

                else:
                    continue

                break

        if patch_index < 0:
            patch_index = labels_group.childCount() - 1

        self.active_label.setSelected(False)

        self.active_label = labels_group.child(patch_index)
        self.active_label.setSelected(True)

    def focus_region(self, label: QTreeWidgetItem):
        if not self._commited:
            self.commit()

        if not isinstance(label, LabelItem):
            return

        self.active_label = label
        current_center = self.active_label.center

        self.viewer.dims.order = tuple(range(self.viewer.dims.ndim))
        self.viewer.camera.center = (
            *self.viewer.camera.center[:-self.viewer.dims.ndisplay],
            *current_center
        )
        self.viewer.camera.zoom = 1

        active_layer_channel = self.active_label.parent().parent()
        self.active_layers_group = active_layer_channel.parent()
        self.active_group = self.active_layers_group.parent()

        for layer in self.viewer.layers:
            layer.visible = False

        self.active_group.visible = True

    def fix_labels(self):
        if (not self.active_layers_group or not self.active_label
           or not self.active_label):
            return

        input_filename = self.active_layers_group.source_data
        data_group = self.active_layers_group.data_group

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
            lazy_data = lazy_data[ts.d[:][self.active_label.position].translate_to[0]]

        elif isinstance(input_filename, MultiScaleData):
            lazy_data = np.copy(input_filename[0][self.active_label.position])
        else:
            lazy_data = np.copy(input_filename[self.active_label.position])

        self.viewer.add_labels(
            lazy_data,
            name="Labels edit",
            blending="translucent_no_depth",
            opacity=0.7,
            translate=[ax_roi.start for ax_roi in self.active_label.position]
        )
        self.viewer.layers["Labels edit"].bounding_box.visible = True

        self._commited = False
        self.commit_btn.setEnabled(True)

    def commit(self):
        if self._transaction:
            self._transaction.commit_async()
            self._transaction = None

        if "Labels edit" in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers["Labels edit"])

        segmentation_channel = None
        if self.active_layers_group:
            segmentation_channel = self.active_layers_group.child(0)

        if segmentation_channel:
            segmentation_channel.layer.refresh()

        self._commited = True
        self.commit_btn.setEnabled(False)


class AcquisitionFunction(QWidget):
    def __init__(self, viewer: napari.Viewer,
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

        self.execute_btn = QPushButton("Run all image groups")
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
    def compute_acquisition(dataset_metadata, acquisition_fun,
                            segmentation_out,
                            MC_repetitions=30,
                            max_samples=1000,
                            patch_size=128):
        dl = datautils.get_dataloader(dataset_metadata, patch_size=patch_size,
                                      shuffle=True)

        model_dropout = cellpose_model_init(use_dropout=True)
        model = cellpose_model_init(use_dropout=False)

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
            acquisition_val = u_sp_lab.max()

            # Execute segmentation once to get the expected labels
            seg_out = cellpose_segment(img[0].numpy(), model)
            seg_out = np.where(seg_out, seg_out + segmentation_max, 0)
            segmentation_out[pos_u_lab] = seg_out

            segmentation_max = max(segmentation_max, seg_out.max())

            img_sampling_positions.append(
                LabelItem(acquisition_val, position=pos_u_lab)
            )

            n_samples += 1

            if n_samples >= max_samples:
                break

        return img_sampling_positions

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
            image_group.setSelected(True)
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

            dataset_metadata = {}

            for layer_type in image_group.layer_type:
                if layer_type not in ["images", "sampling mask"]:
                    continue

                dataset_metadata[layer_type] = image_group.metadata[layer_type]

                layer_source_axes = image_group.source_axes[layer_type]
                dataset_metadata[layer_type]["roi"] = [
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

                dataset_metadata[layer_type]["axes"] = spaial_axes

            img_sampling_positions = self.compute_acquisition(
                dataset_metadata,
                acquisition_fun=acquisition_root["labels/acquisition_fun/0"],
                segmentation_out=segmentation_root["labels/segmentation/0"],
                MC_repetitions=MC_repetitions,
                max_samples=max_samples,
                patch_size=patch_size
            )
            self.image_pb.setValue(n + 1)

            if not img_sampling_positions:
                continue

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
                contrast_limits=(0,
                                 max(img_sampling_positions).acquisition_val),
            )

            segmentation_ms_kwargs = dict(
                name=group_name + " segmentation",
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

            new_acquisition_layer = acquisition_loader(
                acquisition_fun_ms,
                **acquisition_fun_ms_kwargs
            )

            if isinstance(new_acquisition_layer, list):
                new_acquisition_layer = new_acquisition_layer[0]

            new_segmentation_layer = segmentation_loader(
                segmentation_ms,
                **segmentation_ms_kwargs
            )

            if isinstance(new_segmentation_layer, list):
                new_segmentation_layer = new_segmentation_layer[0]

            self.image_groups_manager.root_group.add_managed_layer(
                image_group,
                new_acquisition_layer,
                layer_type="acquisition function",
                channel=None,
                source_axes="YX",
                add_to_output=False
            )

            segmentation_channel = \
                self.image_groups_manager.root_group.add_managed_layer(
                    image_group,
                    new_segmentation_layer,
                    layer_type="segmentation",
                    channel=None,
                    source_axes="YX",
                    add_to_output=False
                )

            label_group = LabelGroup()
            label_group.addChildren(img_sampling_positions)

            segmentation_channel.addChild(label_group)

            label_group.setExpanded(False)
            label_group.sortChildren(0, Qt.SortOrder.DescendingOrder)


class MaskGenerator(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.viewer = viewer
        self._active_image_group = None
        self._active_layers_group = None

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

    @property
    def active_layers_group(self):
        return self._active_layers_group

    @active_layers_group.setter
    def active_layers_group(self,
                                    active_layers_group: LayersGroup):
        self._active_layers_group = active_layers_group

        self.generate_mask_btn.setEnabled(self._active_layers_group is not None)
        self.patch_size_spn.setEnabled(self._active_layers_group is not None)

    @property
    def active_image_group(self):
        return self._active_image_group

    @active_image_group.setter
    def active_image_group(self, active_image_group: ImageGroup):
        self._active_image_group = active_image_group

        self.generate_mask_btn.setEnabled(self._active_layers_group is not None)
        self.patch_size_spn.setEnabled(self._active_layers_group is not None)

    def generate_mask_layer(self):
        if (not self.active_layers_group
           or not self.active_layers_group.childCount()
           or not self.active_layers_group.source_axes):
            return

        patch_size = self.patch_size_spn.value()

        source_axes = [
            ax
            for ax in self.active_layers_group.source_axes
            if ax != "C"
        ]

        reference_layer = self.active_layers_group.child(0).layer
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
            name=self.active_layers_group.layer_group_name + " sampling mask",
        )

        self.active_image_group.add_layer(
            new_mask_layer,
            layer_type="sampling mask",
            source_axes=mask_axes,
            add_to_output=False
        )


if __name__ == "__main__":
    import skimage
    viewer = napari.Viewer()

    img = skimage.data.astronaut()
    viewer.add_image(img[..., 0], blending="opaque", colormap="red")
    viewer.add_image(img[..., 1], blending="additive", colormap="green")
    viewer.add_image(img[..., 2], blending="additive", colormap="blue")

    image_groups_manager = ImageGroupsManager(viewer)
    viewer.window.add_dock_widget(image_groups_manager, area='right')

    acquisition_function = AcquisitionFunction(viewer, image_groups_manager)
    viewer.window.add_dock_widget(acquisition_function, area='right')

    labels_manager_widget = LabelsManager(viewer, image_groups_manager)
    viewer.window.add_dock_widget(labels_manager_widget, area='right')

    napari.run()
