from typing import Iterable, Union, Optional
import operator
from pathlib import Path
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QTreeWidgetItem

import numpy as np
import json

import zarr
import dask.array as da
import napari
from napari.layers import Image, Labels, Layer
from napari.layers._multiscale_data import MultiScaleData

from ._utils import get_source_data, validate_name, get_basename, save_zarr


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
        source_axes = list(source_axes)
        source_axes = source_axes[-self.layer.ndim:]
        self.source_axes = "".join(source_axes)

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
        if self.parent() is not None:
            self.parent().updated = True

    @property
    def data_group(self):
        if not self._data_group:
            self._update_source_data()

        return self._data_group

    @data_group.setter
    def data_group(self, data_group):
        self._data_group = data_group
        if self.parent() is not None:
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
        if "C" in source_axes and self.layer.ndim != len(source_axes):
            source_axes = list(source_axes)
            source_axes.remove("C")
            source_axes = "".join(source_axes)

        self._source_axes = source_axes
        self.setText(3, str(self._source_axes))

    @property
    def name(self):
        return self.layer.name

    @name.setter
    def name(self, name: str):
        self.layer.name = name

    @property
    def shape(self):
        return self.layer.data.shape

    @property
    def ndim(self):
        return self.layer.ndim

    @property
    def scale(self):
        return self.layer.scale

    @scale.setter
    def scale(self, new_scale: Iterable[float]):
        self.layer.scale = new_scale

    @property
    def translate(self):
        return self.layer.translate

    @translate.setter
    def translate(self, new_translate: Iterable[float]):
        self.layer.translate = new_translate

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
            if self.child(idx).ndim == len(self._source_axes):
                self.child(idx).source_axes = self._source_axes
            else:
                self.child(idx).source_axes = self._source_axes_no_channels

    def _update_source_data(self):
        if self.childCount():
            self._source_data = self.child(0).source_data
            self._data_group = self.child(0).data_group

            if (not isinstance(self._source_data, (str, Path))
               and self.childCount() > 1):
                if len(self._source_axes) == self.child(0).ndim:
                    merge_fun = np.concatenate
                else:
                    merge_fun = np.stack

                layers_channels = (
                    (self.child(idx).channel, self.child(idx))
                    for idx in range(self.childCount())
                )

                self._source_data = merge_fun(
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
        if self.parent() is not None:
            layers_group_name = validate_name(
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
            shape = list(self.child(0).shape)

            if "C" in self._source_axes:
                channel_axis = self._source_axes.index("C")
                if len(self._source_axes) == self.child(0).ndim:
                    shape[channel_axis] = (shape[channel_axis]
                                           * self.childCount())
                else:
                    shape.insert(channel_axis, self.childCount())

        return shape

    @property
    def scale(self):
        if not self.childCount():
            n_channels = len(self.source_axes) if self.source_axes else 0
            scale = [1] * n_channels

        else:
            scale = list(self.child(0).scale)

            if "C" in self._source_axes:
                channel_axis = self._source_axes.index("C")
                if len(self._source_axes) != self.child(0).ndim:
                    scale.insert(channel_axis, 1)

            scale = tuple(scale)

        return scale

    @property
    def translate(self):
        if not self.childCount():
            n_channels = len(self.source_axes) if self.source_axes else 0
            translate = [0] * n_channels

        else:
            translate = list(self.child(0).translate)
            if "C" in self._source_axes:
                channel_axis = self._source_axes.index("C")
                if len(self._source_axes) != self.child(0).ndim:
                    translate.insert(channel_axis, 0)

            translate = tuple(translate)

        return translate

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

        if self.parent() is not None and self.parent().parent() is not None:
            self.parent().parent().remove_managed_layer_channel(layer_channel)

        self._update_source_axes()
        self.updated = True

    def takeChildren(self):
        children = super(LayersGroup, self).takeChildren()
        for child in children:
            if isinstance(child, LayerChannel):
                self.remove_layer(child)

        return children

    def takeChild(self, index: int):
        child = super(LayersGroup, self).takeChild(index)
        if isinstance(child, LayerChannel):
            self.remove_layer(child)

        return child

    def removeChild(self, child: QTreeWidgetItem):
        if isinstance(child, LayerChannel):
            self.remove_layer(child)

        super(LayersGroup, self).removeChild(child)

    def addChild(self, child: QTreeWidgetItem):
        if (isinstance(child, LayerChannel)
           and self.parent() is not None
           and self.parent().parent() is not None):
            self.parent().parent().add_managed_layer_channel(child)

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
        if self.parent() is not None:
            group_name = self.parent().group_name
        else:
            group_name = "unset"

        output_filename = output_dir / (group_name + ".zarr")

        source_data = self.source_data

        name = get_basename(self.layers_group_name)

        is_multiscale = isinstance(source_data, MultiScaleData)
        is_multiscale |= (
            isinstance(source_data, list)
            and all(map(lambda src_lvl:
                        isinstance(src_lvl, (zarr.Array, da.core.Array)),
                        source_data))
        )

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

        self._labels_group = None

        self.layers_groups_names = set()

        super().__init__()

        self.group_name = group_name
        self.group_dir = group_dir

    @property
    def group_name(self):
        return self._group_name

    @group_name.setter
    def group_name(self, group_name: str):
        if self.parent() is not None:
            group_name = validate_name(
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

    @property
    def labels_group(self):
        return self._labels_group

    @labels_group.setter
    def labels_group(self, labels_group: Labels):
        self._labels_group = labels_group

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

    def takeChild(self, index: int):
        child = super(ImageGroup, self).takeChild(index)
        if isinstance(child, LayersGroup):
            child.takeChildren()

        return child

    def takeChildren(self):
        children = super(ImageGroup, self).takeChildren()
        for child in children:
            if isinstance(child, LayersGroup):
                child.takeChildren()

        return children

    def removeChild(self, child: QTreeWidgetItem):
        if isinstance(child, LayersGroup):
            child.takeChildren()

        super(ImageGroup, self).removeChild(child)

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
            group_name = "unset"

            if child.group_name:
                group_name = child.group_name

            group_name = validate_name(self.group_names, "", group_name)

            child.group_name = group_name

        super(ImageGroupRoot, self).addChild(child)

    def removeChild(self, child: QTreeWidgetItem):
        if isinstance(child, ImageGroup):
            if child.group_name in self.group_names:
                self.group_names.remove(child.group_name)

            child.takeChildren()

        super(ImageGroupRoot, self).removeChild(child)

    def takeChild(self, index: int):
        child = super(ImageGroupRoot, self).takeChild(index)

        if isinstance(child, ImageGroup):
            self.group_names.remove(child.group_name)
            child.takeChildren()

        return child

    def takeChildren(self):
        children = super(ImageGroupRoot, self).takeChildren()
        for child in children:
            if isinstance(child, ImageGroup):
                child.takeChildren()

        self.group_names.clear()

        return children

    def add_managed_layer_channel(self, layer_channel: LayerChannel):
        layer = layer_channel.layer
        if layer not in self.managed_layers:
            self.managed_layers[layer] = []

        self.managed_layers[layer].append(layer_channel)

        viewer = napari.current_viewer()
        viewer.layers.events.removed.connect(
            self.remove_managed_layer
        )

    def remove_managed_layer_channel(self,
                                     removed_layer_channel: LayerChannel):
        if removed_layer_channel.layer not in self.managed_layers:
            return

        self.managed_layers[removed_layer_channel.layer].remove(
            removed_layer_channel
        )

    def remove_managed_layer(self, event):
        removed_layer = event.value
        if removed_layer not in self.managed_layers:
            return

        for layer_channel in self.managed_layers[removed_layer]:
            layers_group = layer_channel.parent()
            if layers_group is None:
                continue

            image_group = layers_group.parent()

            layers_group.removeChild(layer_channel)

            if not layers_group.childCount():
                image_group.removeChild(layers_group)

            if not image_group.childCount():
                self.removeChild(image_group)

            if image_group.labels_group:
                if image_group.labels_group.parent():
                    image_group.labels_group.parent().removeChild(
                        image_group.labels_group
                    )

                image_group.labels_group = None

        if not self.managed_layers[removed_layer]:
            self.managed_layers.pop(removed_layer)

        self.setSelected(True)


class PropertiesEditor:
    def __init__(self):
        self._active_image_group: Union[None, ImageGroup] = None
        self._active_layers_group: Union[None, LayersGroup] = None
        self._active_layer_channel: Union[None, LayerChannel] = None

        super().__init__()

    @property
    def active_image_group(self):
        return self._active_image_group

    @active_image_group.setter
    def active_image_group(self, active_image_group: Union[ImageGroup, None]):
        self._active_image_group = active_image_group
        self._active_layers_group = None
        self._active_layer_channel = None

    @property
    def active_layers_group(self):
        return self._active_layers_group

    @active_layers_group.setter
    def active_layers_group(self,
                            active_layers_group: Union[LayersGroup, None]):
        self._active_layers_group = active_layers_group
        self._active_layer_channel = None

        if self._active_image_group is None and self._active_layers_group:
            self._active_image_group = self._active_layers_group.parent()

    @property
    def active_layer_channel(self):
        return self._active_layer_channel

    @active_layer_channel.setter
    def active_layer_channel(self,
                             active_layer_channel: Union[LayerChannel, None]):
        self._active_layer_channel = active_layer_channel

        if self._active_layers_group is None and self._active_layer_channel:
            self._active_layers_group = self._active_layer_channel.parent()

        if self._active_image_group is None and self._active_layers_group:
            self._active_image_group = self._active_layers_group.parent()


class ImageGroupEditor(PropertiesEditor):
    def __init__(self):
        super().__init__()

        self._group_name = None
        self._layers_group_name = None
        self._edit_axes = None
        self._edit_scale = None
        self._edit_translate = None
        self._use_as_input = None
        self._use_as_sampling = None
        self._edit_channel = None
        self._output_dir = None

    def update_output_dir(self, output_dir: Optional[Union[Path, str]] = None):
        if output_dir:
            self._output_dir = str(output_dir)
        elif self._active_image_group is not None:
            self._output_dir = str(self._active_image_group.group_dir)

        if self._active_image_group is None:
            return

        if self._output_dir.lower() in ("unset", "none", ""):
            self._output_dir = None

        if self._active_image_group.group_dir != self._output_dir:
            self._active_image_group.group_dir = self._output_dir

    def update_group_name(self, group_name: Optional[str] = None):
        if not self._active_image_group:
            return

        if group_name is not None:
            self._group_name = group_name

        if self._active_image_group.group_name != self._group_name:
            self._active_image_group.group_name = self._group_name

    def update_channels(self, channel: Optional[int] = None):
        if not self._active_layer_channel or not self._active_layers_group:
            return

        if channel:
            self._edit_channel = channel

        prev_channel = self._active_layer_channel.channel
        if prev_channel != self._edit_channel:
            self._active_layers_group.move_channel(prev_channel,
                                                   self._edit_channel)

    def update_source_axes(self, source_axes: str):
        if not self._active_layers_group and not self._active_layer_channel:
            return

        self._edit_axes = source_axes

        if self._active_layers_group:
            if self._active_layers_group.source_axes != self._edit_axes:
                self._active_layers_group.source_axes = self._edit_axes

        if self._active_layer_channel:
            if self._active_layer_channel.source_axes != self._edit_axes:
                self._active_layer_channel.source_axes = self._edit_axes

        display_source_axes = self._edit_axes.lower()

        viewer = napari.current_viewer()
        if display_source_axes != "".join(viewer.dims.axis_labels).lower():
            if ("c" in display_source_axes
               and len(viewer.dims.axis_labels) != len(display_source_axes)):
                display_source_axes = list(display_source_axes)
                display_source_axes.remove("c")
                display_source_axes = "".join(display_source_axes)

            viewer.dims.axis_labels = tuple(display_source_axes)

    def update_layers_group_name(self,
                                 layers_group_name: Optional[str] = None):
        if not self._active_layers_group or not self._active_image_group:
            return False

        if layers_group_name:
            self._layers_group_name = layers_group_name

        if (self._active_layers_group.layers_group_name
           != self._layers_group_name):
            dst_layers_group = self._active_image_group.getLayersGroup(
                self._layers_group_name
            )

            if dst_layers_group:
                self._active_layers_group =\
                    self._active_image_group.move_layer(
                        self._active_layers_group,
                        dst_layers_group,
                        layer_channel=self._active_layer_channel
                    )
            else:
                self._active_layers_group.layers_group_name =\
                    self._layers_group_name

            self._active_layer_channel = None

            return True

        return False

    def update_use_as_input(self, use_it: bool):
        if not self._active_layers_group:
            return

        self._use_as_input = use_it

        layers_group_idx = self.active_image_group.indexOfChild(
            self._active_layers_group
        )

        if self._use_as_input:
            self._active_image_group.input_layers_group = layers_group_idx

        elif self._active_image_group.input_layers_group == layers_group_idx:
            self._active_image_group.input_layers_group = None

    def update_use_as_sampling(self, use_it: bool):
        if not self._active_layers_group:
            return

        self._use_as_sampling = use_it

        layers_group_idx = self.active_image_group.indexOfChild(
            self._active_layers_group
        )

        if self._use_as_sampling:
            self._active_image_group.sampling_mask_layers_group =\
                 layers_group_idx

        elif (self._active_image_group.sampling_mask_layers_group
              == layers_group_idx):
            self._active_image_group.sampling_mask_layers_group = None

    def update_scale(self, scale: Iterable[float]):
        if (not self._active_layers_group or not self._active_layers_group
           or not self._active_layer_channel):
            return

        self._edit_scale = scale
        self._active_layer_channel.scale = self._edit_scale

    def update_translate(self, translate: Iterable[float]):
        if (not self._active_layers_group or not self._active_layers_group
           or not self._active_layer_channel):
            return

        self._edit_translate = translate
        self._active_layer_channel.translate = self._edit_translate


class LayerScaleEditor(PropertiesEditor):
    def __init__(self):
        super().__init__()
        self._edit_scale = None

    def update_scale(self, new_scale: Optional[Iterable[float]] = None):
        if not self._active_layers_group and not self._active_layer_channel:
            return

        if new_scale is not None:
            self._edit_scale = new_scale

        if not all(map(operator.eq,
                       self._active_layer_channel.scale,
                       self._edit_scale)):
            self._active_layer_channel.scale = self._edit_scale


class MaskGenerator(PropertiesEditor):
    def __init__(self):
        super().__init__()
        self._patch_sizes = None
        self._mask_axes = None
        self._im_shape = None
        self._im_scale = None
        self._im_translate = None
        self._im_source_axes = None

    def _update_reference_info(self):
        if (self._active_image_group is None
           or self._active_image_group.input_layers_group is None
           or not self._active_image_group.child(
               self._active_image_group.input_layers_group).childCount()):
            return False

        self._active_layers_group = self._active_image_group.child(
            self._active_image_group.input_layers_group
        )

        self._active_layer_channel = self._active_layers_group.child(0)

        im_shape = self._active_layer_channel.shape
        im_scale = self._active_layer_channel.scale
        im_translate = self._active_layer_channel.translate
        im_source_axes = "".join([
            ax
            for ax in self._active_layers_group.source_axes
            if (ax != "C"
                or len(self._active_layers_group.source_axes) == len(im_shape))
        ])

        self._mask_axes = "".join([
            ax
            for ax, ax_s in zip(im_source_axes, im_shape)
            if ax != "C" and ax_s > 1
        ])

        (self._im_source_axes,
         self._im_shape,
         self._im_scale,
         self._im_translate) = list(zip(*filter(
             lambda ax_props: ax_props[0] in self._mask_axes,
             zip(im_source_axes, im_shape, im_scale, im_translate)
             )))

        self._patch_sizes = [
                min(128, ax_s)
                for ax_s in self._im_shape
            ]
        return True

    def generate_mask_layer(self):
        if (self._active_image_group is None
           or self._active_image_group.input_layers_group is None
           or not self._active_image_group.child(
               self._active_image_group.input_layers_group).childCount()):
            return None

        self._active_layers_group = self._active_image_group.child(
            self._active_image_group.input_layers_group
        )

        self._active_layer_channel = self._active_layers_group.child(0)
        masks_group_name = "mask"

        mask_shape = [
            max(1, ax_s // ax_ps)
            for ax_ps, ax_s, ax in zip(self._patch_sizes, self._im_shape,
                                       self._im_source_axes)
            if ax in self._mask_axes
        ]

        mask_translate = tuple(map(
            lambda ax_ps, ax_s, ax_scl, ax_trans:
            ax_trans + ((ax_scl * (ax_ps - 1) / 2.0)
                        if ax_s // ax_ps >= 1 else 0),
            self._patch_sizes, self._im_shape, self._im_scale,
            self._im_translate
        ))

        mask_scale = tuple(map(
            lambda ax_ps, ax_s, ax_scl:
            (ax_scl * ax_ps)
            if ax_s // ax_ps >= 1 else ax_scl,
            self._patch_sizes, self._im_shape, self._im_scale
        ))

        if self._active_image_group.group_dir:
            mask_output_filename = (self._active_image_group.group_dir
                                    / (self._active_image_group.group_name
                                       + ".zarr"))
            mask_root = save_zarr(
                mask_output_filename,
                data=None,
                shape=mask_shape,
                chunk_size=True,
                name=masks_group_name,
                dtype=np.uint8,
                is_label=True,
                is_multiscale=True
            )

            mask_grp = mask_root[f"labels/{masks_group_name}/0"]
        else:
            mask_grp = np.zeros(mask_shape, dtype=np.uint8)
            mask_output_filename = None

        viewer = napari.current_viewer()
        new_mask_layer = viewer.add_labels(
            data=mask_grp,
            scale=mask_scale,
            translate=mask_translate,
            name=(self._active_layers_group.layers_group_name
                  + " " + masks_group_name)
        )

        masks_layers_group = self._active_image_group.getLayersGroup(
            masks_group_name
        )
        if masks_layers_group is None:
            masks_layers_group = self._active_image_group.add_layers_group(
                masks_group_name,
                source_axes=self._mask_axes,
                use_as_sampling_mask=True
            )

        masks_layer_channel = masks_layers_group.add_layer(new_mask_layer)

        if mask_output_filename:
            masks_layer_channel.source_data = str(mask_output_filename)
            masks_layer_channel.data_group = f"labels/{masks_group_name}/0"

        return new_mask_layer

    def set_patch_size(self, patch_sizes: Union[int, Iterable[int]]):
        if self._mask_axes is None:
            return

        if isinstance(patch_sizes, int):
            patch_sizes = [patch_sizes] * len(self._mask_axes)

        self._patch_sizes = patch_sizes

    @property
    def active_image_group(self):
        return super().active_image_group

    @active_image_group.setter
    def active_image_group(self, active_image_group: Union[ImageGroup, None]):
        if (active_image_group != self._active_image_group):
            self._patch_sizes = None
            self._mask_axes = None
            self._im_shape = None
            self._im_scale = None
            self._im_translate = None
            self._im_source_axes = None

        super(MaskGenerator, type(self)).active_image_group\
                                        .fset(self, active_image_group)

        if self._patch_sizes is None:
            self._update_reference_info()


class ImageGroupsManager:
    def __init__(self, default_axis_labels: str = "TZYX"):

        self._active_layer_channel: Union[None, LayerChannel] = None
        self._active_layers_group: Union[None, LayersGroup] = None
        self._active_image_group: Union[None, ImageGroup] = None

        self.groups_root = ImageGroupRoot()

        viewer = napari.current_viewer()

        ndims = viewer.dims.ndisplay
        extra_dims = ndims - len(default_axis_labels)
        if extra_dims <= 0:
            axis_labels = default_axis_labels[extra_dims:]
        else:
            axis_labels = ("".join(map(str, range(len(default_axis_labels),
                                                  ndims)))
                           + default_axis_labels)

        viewer.dims.axis_labels = list(axis_labels)

        self.image_groups_editor = ImageGroupEditor()
        self.layer_scale_editor = LayerScaleEditor()
        self.mask_generator = MaskGenerator()

        super().__init__()

        self._selected_items = []

    def set_active_item(self,
                        item: Optional[
                            Union[QTreeWidgetItem, Iterable[QTreeWidgetItem]]
                            ] = None):
        if isinstance(item, list):
            if not len(item):
                return

            self._selected_items = item

        elif item is not None:
            self._selected_items = [item]

        item = self._selected_items[-1] if len(self._selected_items) else None

        self._active_layer_channel = None
        self._active_layers_group = None
        self._active_image_group = None

        if isinstance(item, LayerChannel):
            self._active_layer_channel = item

        elif isinstance(item, LayersGroup):
            self._active_layers_group = item

        elif isinstance(item, ImageGroup):
            self._active_image_group = item

        self.layer_scale_editor.active_layer_channel =\
            self._active_layer_channel

        if self._active_layer_channel:
            self._active_layers_group = self._active_layer_channel.parent()

        if self._active_layers_group:
            self._active_image_group = self._active_layers_group.parent()

        self.mask_generator.active_image_group = self._active_image_group

        self.image_groups_editor.active_image_group =\
            self._active_image_group

        self.image_groups_editor.active_layers_group =\
            self._active_layers_group

        self.image_groups_editor.active_layer_channel =\
            self._active_layer_channel

    def get_active_item(self):
        return self._selected_items

    def focus_active_item(self, item: Union[QTreeWidgetItem,
                                            Iterable[QTreeWidgetItem]]):
        if isinstance(item, list):
            if not len(item):
                return

            item = item[0]

        if not item:
            return

        viewer = napari.current_viewer()
        viewer.layers.selection.clear()
        for layer in viewer.layers:
            layer.visible = False

        if isinstance(item, (LayerChannel, LayersGroup, ImageGroup)):
            item.visible = True
            item.selected = True
            item.setExpanded(True)

    def update_group(self):
        viewer = napari.current_viewer()
        selected_layers = viewer.layers.selection

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

            viewer = napari.current_viewer()
            viewer.layers.selection.clear()
            viewer.layers.selection = viewer.layers.selection.union(
                layers_set
            )

            self.add_layers_to_group()

        self._active_image_group.setExpanded(True)

    def create_group(self):
        self._active_image_group = ImageGroup()
        self.groups_root.addChild(self._active_image_group)
        self.set_active_item(self._active_image_group)

        self._active_image_group.setExpanded(True)

        self.update_group()

    def create_layers_group(self):
        if self._active_image_group is None:
            return

        viewer = napari.current_viewer()
        active_source_axes = "".join(viewer.dims.axis_labels).upper()

        self._active_layers_group = self._active_image_group.add_layers_group(
            source_axes=active_source_axes
        )
        self.set_active_item(self._active_layers_group)

        self._active_layers_group.setExpanded(True)

    def add_layers_to_group(self):
        viewer = napari.current_viewer()
        selected_layers = sorted(
            map(lambda layer: (layer.name, layer),
                viewer.layers.selection)
        )

        selected_layers = list(zip(*selected_layers))

        if not selected_layers:
            return

        for layer in selected_layers[1]:
            self._active_layer_channel = self._active_layers_group.add_layer(
                layer=layer
            )
            self.set_active_item(self._active_layer_channel)
            self._active_layer_channel.setExpanded(True)

            if (self._active_image_group.group_name is None
               or "unset" in self._active_image_group.group_name):
                self._active_image_group.group_name =\
                    self._active_layer_channel.name

    def remove_layer(self):
        self._active_layers_group.removeChild(self._active_layer_channel)
        self._active_layer_channel = None

        self.layer_scale_editor.active_layer_channel = None
        self.image_groups_editor.active_layer_channel = None

    def remove_layers_group(self):
        self._active_image_group.removeChild(self._active_layers_group)

        self._active_layer_channel = None
        self._active_layers_group = None

        self.layer_scale_editor.active_layer_channel = None
        self.image_groups_editor.active_layer_channel = None

    def remove_group(self):
        self.groups_root.removeChild(self._active_image_group)

        self._active_layer_channel = None
        self._active_layers_group = None
        self._active_image_group = None

        self.layer_scale_editor.active_layer_channel = None
        self.image_groups_editor.active_layer_channel = None

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
                try:
                    fp.write(json.dumps(self._active_image_group.metadata))
                except TypeError:
                    raise ValueError("Some images in this group are arrays, "
                                     "save them as zarr files to generate the"
                                     "metadata file.")
