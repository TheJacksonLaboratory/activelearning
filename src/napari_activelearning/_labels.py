from typing import List, Iterable, Union, Optional
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QTreeWidgetItem

import numpy as np
import tensorstore as ts
import zarr

import napari
from napari.layers import Layer
from napari.layers._multiscale_data import MultiScaleData

from ._layers import ImageGroup, LayersGroup, LayerChannel
from ._utils import update_labels


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

        self._center = list(map(
            lambda tl, br:
            (tl + br) / 2 if tl is not None and br is not None else 0.5,
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

    def __gt__(self, other_label):
        return self._acquisition_val > other_label._acquisition_val


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

    def add_managed_label_group(self, label_group: LabelGroup):
        layer = label_group.layer_channel.layer
        layers_group = label_group.layer_channel.parent()
        if layers_group is not None:
            image_group = layers_group.parent()
        else:
            image_group = None

        self.managed_layers[layer] = (label_group, image_group)

        viewer = napari.current_viewer()
        viewer.layers.events.removed.connect(
            self.remove_managed_layer
        )

    def remove_managed_label_group(self, label_group: LabelGroup):
        layer = label_group.layer_channel.layer

        if layer in self.managed_layers:
            (label_group,
             image_group) = self.managed_layers.pop(layer)

            if image_group is not None:
                image_group.labels_group = None

        self.setSelected(True)

    def remove_managed_layer(self, event):
        removed_layer = event.value

        label_group_list = self.managed_layers.get(removed_layer, [])
        for label_group in label_group_list:
            self.removeChild(label_group)

        self.setSelected(True)

    def addChild(self, child: QTreeWidgetItem):
        if isinstance(child, LabelGroup):
            if child.layer_channel:
                self.add_managed_label_group(child)

        super(LabelGroupRoot, self).addChild(child)

    def addChildren(self, children: Iterable[QTreeWidgetItem]):
        for child in children:
            if isinstance(child, LabelGroup):
                if child.layer_channel:
                    self.add_managed_label(child.layer_channel.layer, child)

        super(LabelGroupRoot, self).addChildred(children)

    def removeChild(self, child: QTreeWidgetItem):
        if isinstance(child, LabelGroup) and child.layer_channel:
            self.remove_managed_label_group(child)

        super(LabelGroupRoot, self).removeChild(child)

        self.setSelected(True)

    def takeChild(self, index: int):
        child = super(LabelGroupRoot, self).takeChild(index)
        if isinstance(child, LabelGroup) and child.layer_channel:
            self.remove_managed_label_group(child)

        self.setSelected(True)

        return child

    def takeChildren(self):
        children = super(LabelGroupRoot, self).takeChildren()
        for child in children:
            if isinstance(child, LabelGroup) and child.layer_channel:
                self.remove_managed_label_group(child)

        self.setSelected(True)

        return children


class LabelsManager:
    def __init__(self):
        super().__init__()

        self.labels_group_root = LabelGroupRoot()

        self._active_label: Union[None, LabelItem] = None
        self._active_label_group: Union[None, LabelGroup] = None
        self._active_layer_channel: Union[None, LayerChannel] = None
        self._active_layers_group: Union[None, LayersGroup] = None
        self._active_image_group: Union[None, ImageGroup] = None

        self._transaction = None
        self._active_edit_layer: Union[None, Layer] = None

        self._requires_commit = False

        viewer = napari.current_viewer()
        viewer.layers.events.removed.connect(
            self.commit
        )

    def _load_label_data(self, input_filename, data_group=None):
        if isinstance(input_filename, (Path, str)):
            label_data_grp = zarr.open(Path(input_filename) / data_group)
            label_data = label_data_grp[self._active_label.position]

        elif isinstance(input_filename, MultiScaleData):
            label_data = np.array(
                input_filename[0][self._active_label.position]
            )
        else:
            label_data = np.array(input_filename[self._active_label.position])

        return label_data

    def _write_label_data(self, label_data: Optional[np.ndarray]):
        if isinstance(self._transaction, ts.Transaction):
            self._transaction.commit_async()
        elif (self._active_label.position is not None
                and self._active_layers_group is not None):
            input_filename = self._active_layers_group.source_data
            data_group = self._active_layers_group.data_group

            if isinstance(input_filename, (Path, str)):
                if ".zarr" in str(input_filename):
                    segmentation_channel_group = input_filename
                else:
                    raise ValueError("File format not supported for "
                                     "writing labels.")

                segmentation_channel_group = zarr.open(
                    segmentation_channel_group,
                    mode="r+"
                )

                if data_group is not None:
                    data_group_base = str(Path(*Path(data_group).parts[:-1]))

                down_scales = len(
                    segmentation_channel_group[data_group_base].keys()
                )

                segmentation_channel_data = [
                    segmentation_channel_group[f"{data_group_base}/{grp}"]
                    for grp in range(down_scales)
                ]

                update_labels(
                    segmentation_channel_group[f"{data_group_base}"],
                    label_data if label_data is not None else set()
                )

            elif isinstance(input_filename, MultiScaleData):
                segmentation_channel_data =\
                    self._active_layers_group.source_data
            else:
                segmentation_channel_data = [
                    self._active_layers_group.source_data
                ]

            for s_scl, seg_data in enumerate(segmentation_channel_data):
                curr_position = tuple(
                    list(self._active_label.position[:-2])
                    + [slice(pos_sel.start // 2**s_scl,
                             pos_sel.stop // 2**s_scl)
                       for pos_sel in self._active_label.position[-2:]]
                )
                if label_data is not None:
                    seg_data[curr_position] =\
                        label_data[..., ::2**s_scl, ::2**s_scl]
                else:
                    seg_data[curr_position] = 0

    def add_labels(self, layer_channel: LayerChannel,
                   labels: Iterable[LabelItem]):
        new_label_group = LabelGroup(layer_channel)
        new_label_group.addChildren(labels)

        self.labels_group_root.addChild(new_label_group)

        new_label_group.setExpanded(False)
        new_label_group.sortChildren(0, Qt.SortOrder.DescendingOrder)

        layer_channel.layer.mouse_double_click_callbacks.append(
            self.focus_and_edit_region
        )

        return new_label_group

    def remove_labels(self):
        if self._active_label is None and self._active_label_group is None:
            return

        # Set the content of the current label to 0
        self._write_label_data(None)

        self._active_label_group.removeChild(self._active_label)

        if not self._active_label_group.childCount():
            self.labels_group_root.removeChild(self._active_label_group)
            self.labels_group_root.setSelected(True)
            self._active_label_group = None

        else:
            self._active_label_group.setSelected(True)

        self._active_label = None
        self._requires_commit = False
        self.commit()

    def remove_labels_group(self):
        if self._active_label_group is None:
            return

        self._active_layers_group = self._active_label_group.layer_channel
        while (self._active_label_group is not None
               and self._active_label_group.childCount()):
            self._active_label = self._active_label_group.child(0)
            self.remove_labels()

        self.labels_group_root.setSelected(True)
        self._active_label_group = None

        self._requires_commit = False
        self.commit()

    def navigate(self, delta_patch_index=0, delta_image_index=0):
        self.commit()

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

            self._active_image_group = None
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

        self._active_label.setSelected(False)
        self._active_label = self._active_label_group.child(patch_index)
        self._active_label.setSelected(True)
        self.focus_region(self._active_label)

    def focus_region(self, label: Optional[QTreeWidgetItem] = None,
                     edit_focused_label: bool = False):
        if self._requires_commit:
            self.commit()

        if isinstance(label, list) and len(label):
            label = label[0]
        elif not isinstance(label, (LabelItem, LabelGroup)):
            label = None

        self._active_label_group = None
        self._active_label = None

        self._active_image_group = None
        self._active_layers_group = None
        self._active_layer_channel = None

        if isinstance(label, LabelGroup):
            self._active_label_group = label

        if isinstance(label, LabelItem):
            self._active_label = label
            self._active_label_group = self._active_label.parent()

        else:
            return

        self._active_layer_channel = self._active_label_group.layer_channel

        if self._active_label_group is not None:
            self._active_layers_group = self._active_layer_channel.parent()

        if self._active_layers_group is not None:
            self._active_image_group = self._active_layers_group.parent()

        current_center = [
            pos * ax_scl
            for pos, ax_scl in zip(self._active_label.center,
                                   self._active_layer_channel.layer.scale)
        ]

        viewer = napari.current_viewer()
        viewer.dims.order = tuple(range(viewer.dims.ndim))
        viewer.camera.center = current_center
        viewer.dims.current_step = tuple(map(int, current_center))

        # TODO: Only make the labels visible, keeping all the labels that are visible as they are
        # for layer in viewer.layers:
        #     layer.visible = False

        # self._active_image_group.visible = True

        if edit_focused_label:
            self.edit_labels()

    def focus_and_edit_region(self, layer, event):
        clicked_label = None
        curr_pos = layer.world_to_data(event.position)

        for label_group in self.labels_group_root.managed_layers.get(layer,
                                                                     []):
            for label in map(lambda idx: label_group.child(idx),
                             range(label_group.childCount())):
                if not isinstance(label, LabelItem):
                    continue

                if all(ax_pos.start <= ax_coord < ax_pos.stop
                       for ax_pos, ax_coord in zip(label.position, curr_pos)):
                    clicked_label = label
                    break

            else:
                continue

            break

        else:
            return

        self.labels_group_root.setSelected(False)
        for label in map(lambda idx: self.labels_group_root.child(idx),
                         range(self.labels_group_root.childCount())):
            label.setSelected(False)

        clicked_label.setSelected(True)
        self.focus_region(clicked_label, edit_focused_label=True)

    def edit_labels(self):
        if (not self._active_layers_group or not self._active_label
           or not self._active_label):
            return False

        input_filename = self._active_layers_group.source_data
        data_group = self._active_layers_group.data_group

        label_data = self._load_label_data(input_filename, data_group)

        viewer = napari.current_viewer()
        self._active_edit_layer = viewer.add_labels(
            label_data,
            name="Labels edit",
            blending="translucent_no_depth",
            opacity=0.7,
            translate=[
                ax_roi.start * ax_scl * ax_sl_scl
                for ax_roi, ax_sl_scl, ax_scl in zip(
                    self._active_label.position,
                    self._active_layer_channel.selected_level_scale,
                    self._active_layer_channel.layer.scale
                )
            ],
            # scale=self._active_layer_channel.layer.scale
            scale=[
                ax_scl * ax_sl_scl
                for ax_sl_scl, ax_scl in zip(
                    self._active_layer_channel.selected_level_scale,
                    self._active_layer_channel.layer.scale
                )
            ]
        )
        viewer.layers["Labels edit"].bounding_box.visible = True
        self._active_layer_channel.layer.visible = False

        self._requires_commit = True
        return True

    def commit(self):
        edit_data = None

        if self._requires_commit:
            if self._active_edit_layer:
                edit_data = self._active_edit_layer.data

            self._write_label_data(edit_data)

        viewer = napari.current_viewer()
        if (self._active_edit_layer
           and self._active_edit_layer in viewer.layers):
            viewer.layers.remove(self._active_edit_layer)

        if self._active_layer_channel:
            self._active_layer_channel.layer.refresh()
            self._active_layer_channel.visible = True
            self._active_layer_channel.selected = True

        self._transaction = None
        self._active_edit_layer = None
        self._requires_commit = False
