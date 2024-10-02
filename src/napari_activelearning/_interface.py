from typing import Optional, Union, Iterable

from qtpy.QtGui import QIntValidator, QDoubleValidator
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (QWidget, QPushButton, QGridLayout, QVBoxLayout,
                            QLineEdit,
                            QComboBox,
                            QLabel,
                            QFileDialog,
                            QAbstractSpinBox,
                            QSpinBox,
                            QDoubleSpinBox,
                            QCheckBox,
                            QProgressBar,
                            QTreeWidget,
                            QTreeWidgetItem,
                            QAbstractItemView,
                            QScrollArea)

from functools import partial
import math
import numpy as np
import dask.array as da
from napari.layers._multiscale_data import MultiScaleData

from ._acquisition import AcquisitionFunction, TunableMethod
from ._layers import (ImageGroupEditor, ImageGroupsManager, LayerScaleEditor,
                      MaskGenerator,
                      ImageGroup,
                      ImageGroupRoot)
from ._labels import LayersGroup, LayerChannel, LabelsManager


class MultiSpinBox(QWidget):
    sizesChanged = Signal(dict)

    def __init__(self):
        super().__init__()

        self._dtype = int

        self.edit_scale_lyt = QGridLayout()
        self.setLayout(self.edit_scale_lyt)

        self._curr_power_spn_list = []
        self._curr_scale_le_list = []
        self._curr_labels_list = []

        self._sizes = {}

    @property
    def axes(self):
        return self._sizes.keys()

    @axes.setter
    def axes(self, new_axes: str):
        self._sizes = {
            ax: 128
            for ax in new_axes
        }
        self.update_spin_boxes()

    @property
    def sizes(self):
        return self._sizes

    @sizes.setter
    def sizes(self, new_sizes: Union[Iterable, dict]):
        if isinstance(new_sizes, dict):
            self._sizes = new_sizes
            self._axes = new_sizes.keys()

        else:
            self._sizes = {
                ax: ax_s
                for ax, ax_s in zip(self._axes, new_sizes)
            }

            for scale_le, ax_s in zip(self._curr_scale_le_list, self._sizes):
                scale_le.setValue(ax_s)

        self.update_spin_boxes()

    @staticmethod
    def _create_spinbox(ax_s: int):
        power_spn = QSpinBox(
            minimum=0, maximum=16,
            buttonSymbols=QAbstractSpinBox.UpDownArrows
        )
        power_spn.lineEdit().hide()
        power_spn.setValue(int(math.log2(ax_s)))

        scale_le = QLineEdit()
        scale_le.setValidator(QIntValidator(1, 2**16))
        scale_le.setText(str(ax_s))

        return scale_le, power_spn

    def clear_layer_channel(self):
        while self._curr_scale_le_list:
            item = self._curr_scale_le_list.pop()
            self.edit_scale_lyt.removeWidget(item)
            item.deleteLater()

        while self._curr_labels_list:
            item = self._curr_labels_list.pop()
            self.edit_scale_lyt.removeWidget(item)
            item.deleteLater()

        while self._curr_power_spn_list:
            item = self._curr_power_spn_list.pop()
            self.edit_scale_lyt.removeWidget(item)
            item.deleteLater()

    def update_spin_boxes(self):
        self.clear_layer_channel()

        for ax_idx, (ax, ax_s) in enumerate(self._sizes.items()):
            scale_le, power_spn = self._create_spinbox(ax_s)

            self._curr_scale_le_list.append(scale_le)
            self.edit_scale_lyt.addWidget(self._curr_scale_le_list[-1],
                                          ax_idx + 1,
                                          1)
            self._curr_power_spn_list.append(power_spn)
            self.edit_scale_lyt.addWidget(self._curr_power_spn_list[-1],
                                          ax_idx + 1,
                                          2)
            self._curr_labels_list.append(QLabel(ax))
            self.edit_scale_lyt.addWidget(self._curr_labels_list[-1],
                                          ax_idx + 1,
                                          0)

            scale_le.textChanged.connect(
                self._set_patch_size
            )
            power_spn.valueChanged.connect(
                partial(self._modify_size, ax_idx=ax_idx)
            )

        self._set_patch_size()

    def _modify_size(self, scale: int, ax_idx: int = 0):
        self._curr_scale_le_list[ax_idx].setText(str(2 ** scale))

    def _set_patch_size(self):
        axes = self._sizes.keys()
        self._sizes = {
            ax: self._dtype(scale_le.text())
            for ax, scale_le in zip(axes, self._curr_scale_le_list)
        }

        self.sizesChanged.emit(self._sizes)


class MultiDoubleSpinBox(MultiSpinBox):
    def __init__(self):
        super().__init__()
        self._dtype = float

    @staticmethod
    def _create_spinbox(ax_s: float):
        power_spn = QDoubleSpinBox(
            minimum=-16, maximum=16,
            buttonSymbols=QAbstractSpinBox.UpDownArrows
        )
        power_spn.lineEdit().hide()
        power_spn.setValue(math.log2(max(1e-12, ax_s)))

        scale_le = QLineEdit()
        scale_le.setValidator(QDoubleValidator(1e-12, 1e12, 12))
        scale_le.setText(str(ax_s))

        return scale_le, power_spn


class ImageGroupEditorWidget(ImageGroupEditor, QWidget):
    def __init__(self):
        super().__init__()

        self.group_name_le = QLineEdit("None selected")
        self.group_name_le.setEnabled(False)
        self.group_name_le.returnPressed.connect(self.update_group_name)

        self.layers_group_name_cmb = QComboBox()
        self.layers_group_name_cmb.setEditable(True)
        self.layers_group_name_cmb.lineEdit().returnPressed.connect(
            self.update_layers_group_name
        )
        self.layers_group_name_cmb.setEnabled(False)

        self.display_name_lbl = QLabel("None selected")

        self.edit_channel_spn = QSpinBox(minimum=0, maximum=0)
        self.edit_channel_spn.setEnabled(False)
        self.edit_channel_spn.editingFinished.connect(self.update_channels)
        self.edit_channel_spn.valueChanged.connect(self.update_channels)

        self.edit_axes_le = QLineEdit("None selected")
        self.edit_axes_le.setEnabled(False)
        self.edit_axes_le.returnPressed.connect(self.update_source_axes)

        self.output_dir_lbl = QLabel("Output directory:")
        self.output_dir_le = QLineEdit("Unset")
        self.output_dir_dlg = QFileDialog(fileMode=QFileDialog.Directory)
        self.output_dir_btn = QPushButton("...")
        self.output_dir_le.setEnabled(False)
        self.output_dir_btn.setEnabled(False)
        self.output_dir_btn.clicked.connect(self.output_dir_dlg.show)
        self.output_dir_dlg.fileSelected.connect(
            self._update_output_dir_edit
        )
        self.output_dir_le.returnPressed.connect(self.update_output_dir)

        self.use_as_input_chk = QCheckBox("Use as input")
        self.use_as_input_chk.setEnabled(False)
        self.use_as_input_chk.toggled.connect(
            self.update_use_as_input
        )

        self.use_as_sampling_chk = QCheckBox("Use as sampling mask")
        self.use_as_sampling_chk.setEnabled(False)
        self.use_as_sampling_chk.toggled.connect(
            self.update_use_as_sampling
        )

        self.edit_scale_mdspn = MultiDoubleSpinBox()
        self.edit_scale_mdspn.setEnabled(False)
        self.edit_scale_mdspn.sizesChanged.connect(self.update_scale)

        self.edit_translate_mdspn = MultiDoubleSpinBox()
        self.edit_translate_mdspn.setEnabled(False)
        self.edit_translate_mdspn.sizesChanged.connect(self.update_translate)

        show_editor_chk = QCheckBox("Edit group properties")
        show_editor_chk.setChecked(False)
        show_editor_chk.toggled.connect(self._show_editor)

        editor_grid_lyt = QGridLayout()
        editor_grid_lyt.addWidget(QLabel("Group name:"), 0, 0)
        editor_grid_lyt.addWidget(self.group_name_le, 0, 1)
        editor_grid_lyt.addWidget(QLabel("Channels group name:"), 0, 2)
        editor_grid_lyt.addWidget(self.layers_group_name_cmb, 0, 3)
        editor_grid_lyt.addWidget(QLabel("Channel name:"), 1, 0)
        editor_grid_lyt.addWidget(self.display_name_lbl, 1, 1)
        editor_grid_lyt.addWidget(QLabel("Channel:"), 1, 2)
        editor_grid_lyt.addWidget(self.edit_channel_spn, 1, 3)
        editor_grid_lyt.addWidget(QLabel("Axes order:"), 2, 0)
        editor_grid_lyt.addWidget(self.edit_axes_le, 2, 1)
        editor_grid_lyt.addWidget(QLabel("Output directory:"), 3, 0)
        editor_grid_lyt.addWidget(self.output_dir_le, 3, 1, 1, 3)
        editor_grid_lyt.addWidget(self.output_dir_btn, 3, 3)
        editor_grid_lyt.addWidget(self.use_as_input_chk, 4, 0)
        editor_grid_lyt.addWidget(self.use_as_sampling_chk, 4, 1)
        editor_grid_lyt.addWidget(QLabel("Layer scale"), 5, 0)
        editor_grid_lyt.addWidget(self.edit_scale_mdspn, 5, 1)
        editor_grid_lyt.addWidget(QLabel("Layer translate"), 5, 2)
        editor_grid_lyt.addWidget(self.edit_translate_mdspn, 5, 3)

        self.editor_widget = QWidget()
        self.editor_widget.setLayout(editor_grid_lyt)

        editor_lyt = QVBoxLayout()
        editor_lyt.addWidget(show_editor_chk)
        editor_lyt.addWidget(self.editor_widget)

        self.setLayout(editor_lyt)

        self.editor_widget.setVisible(False)

    def _show_editor(self, show: bool):
        self.editor_widget.setVisible(show)

    def _clear_image_group(self):
        self.layers_group_name_cmb.clear()
        self.group_name_le.setText("None selected")

        self.group_name_le.setEnabled(False)
        self.output_dir_le.setEnabled(False)
        self.output_dir_btn.setEnabled(False)

    def _clear_layers_group(self):
        self.edit_axes_le.setText("None selected")

        self.edit_axes_le.setEnabled(False)
        self.layers_group_name_cmb.setEnabled(False)
        self.use_as_input_chk.setEnabled(False)
        self.use_as_sampling_chk.setEnabled(False)
        self.edit_scale_mdspn.axes = ""
        self.edit_translate_mdspn.axes = ""

    def _clear_layer_channel(self):
        self.display_name_lbl.setText("None selected")
        self.edit_channel_spn.setValue(0)
        self.edit_channel_spn.setMaximum(0)
        self.edit_channel_spn.setEnabled(False)
        self.edit_scale_mdspn.setEnabled(False)
        self.edit_translate_mdspn.setEnabled(False)

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

            self.edit_scale_mdspn.sizes = {
                ax: ax_scl
                for ax, ax_scl in zip(self._active_layer_channel.source_axes,
                                      self._active_layer_channel.scale)
                if ax != "C"
            }
            self.edit_scale_mdspn.setEnabled(True)

            self.edit_translate_mdspn.sizes = {
                ax: ax_scl
                for ax, ax_scl in zip(self._active_layer_channel.source_axes,
                                      self._active_layer_channel.translate)
                if ax != "C"
            }
            self.edit_translate_mdspn.setEnabled(True)

    def _update_output_dir_edit(self, path):
        self.output_dir_le.setText(self.output_dir_dlg.selectedFiles()[0])
        self.update_output_dir()

    def update_output_dir(self):
        super().update_output_dir(self.output_dir_le.text())

    def update_group_name(self):
        super().update_group_name(self.group_name_le.text())

    def update_channels(self):
        super().update_channels(self.edit_channel_spn.value())

    def update_source_axes(self):
        super().update_source_axes(self.edit_axes_le.text())

    def update_layers_group_name(self):
        if super().update_layers_group_name(
          self.layers_group_name_cmb.lineEdit().text()):
            self._clear_layer_channel()

    def update_use_as_input(self):
        super().update_use_as_input(self.use_as_input_chk.isChecked())

    def update_use_as_sampling(self):
        super().update_use_as_sampling(self.use_as_sampling_chk.isChecked())

    def update_scale(self):
        super().update_scale(list(self.edit_scale_mdspn.sizes.values()))

    def update_translate(self):
        super().update_translate(
            list(self.edit_translate_mdspn.sizes.values())
        )

    @property
    def active_image_group(self):
        return super().active_image_group

    @active_image_group.setter
    def active_image_group(self, active_image_group: Union[ImageGroup, None]):
        super(ImageGroupEditorWidget, type(self)).active_image_group\
                                                 .fset(self,
                                                       active_image_group)
        self._fill_image_group()

    @property
    def active_layers_group(self):
        return super().active_layers_group

    @active_layers_group.setter
    def active_layers_group(self,
                            active_layers_group: Union[LayersGroup, None]):
        super(ImageGroupEditorWidget, type(self)).active_layers_group\
                                                 .fset(self,
                                                       active_layers_group)
        self._fill_layers_group()

    @property
    def active_layer_channel(self):
        return super().active_layer_channel

    @active_layer_channel.setter
    def active_layer_channel(self,
                             active_layer_channel: Union[LayerChannel, None]):
        super(ImageGroupEditorWidget, type(self)).active_layer_channel\
                                                 .fset(self,
                                                       active_layer_channel)
        self._fill_layer()


class MaskGeneratorWidget(MaskGenerator, QWidget):
    def __init__(self):
        super().__init__()

        self.generate_mask_btn = QPushButton("Create mask")
        self.generate_mask_btn.setToolTip("Create a napari Label layer with a "
                                          "blank mask at the scale selected "
                                          "with `Patch size`")
        self.generate_mask_btn.setEnabled(False)
        self.generate_mask_btn.clicked.connect(self.generate_mask_layer)

        self.patch_sizes_mspn = MultiSpinBox()
        self.patch_sizes_mspn.sizesChanged.connect(self._set_patch_size)
        patch_sizes_scr = QScrollArea()
        patch_sizes_scr.setWidget(self.patch_sizes_mspn)
        patch_sizes_scr.setWidgetResizable(True)
        patch_sizes_scr.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        edit_mask_lyt = QVBoxLayout()
        edit_mask_lyt.addWidget(patch_sizes_scr)
        edit_mask_lyt.addWidget(self.generate_mask_btn)
        self.edit_mask_widget = QWidget()
        self.edit_mask_widget.setLayout(edit_mask_lyt)

        show_editor_chk = QCheckBox("Edit mask properties")
        show_editor_chk.setChecked(False)
        show_editor_chk.toggled.connect(self._show_editor)

        mask_lyt = QVBoxLayout()
        mask_lyt.addWidget(show_editor_chk)
        mask_lyt.addWidget(self.edit_mask_widget)

        self.setLayout(mask_lyt)

        self.edit_mask_widget.setVisible(False)

    def _show_editor(self, show: bool):
        self.edit_mask_widget.setVisible(show)

    def _set_patch_size(self, patch_sizes):
        super().set_patch_size(list(patch_sizes.values()))

    def _update_reference_info(self):
        if super()._update_reference_info():
            self.generate_mask_btn.setEnabled(True)
            self.patch_sizes_mspn.axes = self._mask_axes

        else:
            self.generate_mask_btn.setEnabled(False)
            self.patch_sizes_mspn.axes = ""

    def generate_mask_layer(self):
        self.generate_mask_btn.setEnabled(
            super().generate_mask_layer() is not None
        )


class ImageGroupsManagerWidget(ImageGroupsManager, QWidget):
    def __init__(self, default_axis_labels: str = "TZYX"):
        super().__init__(default_axis_labels)

        # Re-instanciate the following objects with their widget versions.
        self.image_groups_editor = ImageGroupEditorWidget()
        self.mask_generator = MaskGeneratorWidget()

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
            self.focus_active_item
        )
        self.image_groups_tw.itemSelectionChanged.connect(
            self._get_active_item
        )
        self.image_groups_tw.setExpandsOnDoubleClick(False)
        self.image_groups_tw.addTopLevelItem(self.groups_root)
        self.groups_root.setExpanded(True)

        self.new_group_btn = QPushButton("New image group")
        self.new_group_btn.setToolTip("Create a new group. If layers are "
                                      "selected, add these to the new group.")
        self.new_group_btn.clicked.connect(self.create_group)

        self.add_group_btn = QPushButton("Add to group")
        self.add_group_btn.setEnabled(False)
        self.add_group_btn.setToolTip("Add selected layers to current group")
        self.add_group_btn.clicked.connect(self.update_group)

        self.remove_group_btn = QPushButton("Remove group")
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
        self.save_layers_group_btn.clicked.connect(self._save_layers_group)
        self.add_layer_btn.clicked.connect(self.add_layers_to_group)
        self.remove_layer_btn.clicked.connect(self.remove_layer)
        self.save_metadata_btn.clicked.connect(self.dump_dataset_specs)

        manager_lyt = QGridLayout()
        manager_lyt.addWidget(self.new_group_btn, 0, 0)
        manager_lyt.addWidget(self.add_group_btn, 0, 1)
        manager_lyt.addWidget(self.remove_group_btn, 0, 2)

        manager_lyt.addWidget(self.new_layers_group_btn, 1, 0)
        manager_lyt.addWidget(self.remove_layers_group_btn, 1, 1)
        manager_lyt.addWidget(self.save_layers_group_btn, 1, 2)

        manager_lyt.addWidget(self.add_layer_btn, 2, 0)
        manager_lyt.addWidget(self.remove_layer_btn, 2, 1)
        manager_lyt.addWidget(self.save_metadata_btn, 2, 2)

        manager_lyt.addWidget(self.image_groups_editor, 3, 0, 1, 3)
        manager_lyt.addWidget(self.mask_generator, 4, 0, 1, 3)
        manager_lyt.addWidget(self.image_groups_tw, 5, 0, 1, 3)

        self.setLayout(manager_lyt)

    def _save_layers_group(self):
        self.save_layers_group()
        self.save_layers_group_btn.setEnabled(False)

    def _get_active_item(self):
        item_list = list(filter(
            lambda item: isinstance(item, (ImageGroupRoot,
                                           ImageGroup,
                                           LayersGroup,
                                           LayerChannel)),
            self.image_groups_tw.selectedItems()
        ))

        self.set_active_item(item_list)

        self.remove_group_btn.setEnabled(
            self._active_image_group is not None
        )

        self.add_group_btn.setEnabled(
            self._active_image_group is not None
        )

        self.new_layers_group_btn.setEnabled(
            self._active_image_group is not None
        )

        self.save_layers_group_btn.setEnabled(
            self._active_layers_group is not None
            and self._active_image_group is not None
            and self._active_image_group.group_dir is not None
            and isinstance(self._active_layers_group.source_data,
                           (np.ndarray, da.core.Array, MultiScaleData))
        )

        self.remove_layers_group_btn.setEnabled(
            self._active_layers_group is not None
        )

        self.add_layer_btn.setEnabled(
            self._active_layers_group is not None
        )

        self.remove_layer_btn.setEnabled(
            self._active_layer_channel is not None
        )

        self.save_metadata_btn.setEnabled(
            self._active_image_group is not None
            and self._active_image_group.group_dir is not None
        )

    def update_group(self):
        super(ImageGroupsManagerWidget, self).update_group()

        if self._active_image_group:
            self.image_groups_tw.clearSelection()
            self._active_image_group.setSelected(True)

    def create_group(self):
        super(ImageGroupsManagerWidget, self).create_group()

        if self._active_image_group:
            self.image_groups_tw.clearSelection()
            self._active_image_group.setSelected(True)

    def create_layers_group(self):
        super(ImageGroupsManagerWidget, self).create_layers_group()

        if self._active_layers_group:
            self.image_groups_tw.clearSelection()
            self._active_layers_group.setSelected(True)

    def add_layers_to_group(self):
        super(ImageGroupsManagerWidget, self).add_layers_to_group()

        if self._active_layer_channel:
            self.image_groups_tw.clearSelection()
            self._active_layer_channel.setSelected(True)

    def remove_layer(self):
        super(ImageGroupsManagerWidget, self).remove_layer()

        if self._active_layers_group:
            self.image_groups_tw.clearSelection()
            self._active_layers_group.setSelected(True)

    def remove_layers_group(self):
        super(ImageGroupsManagerWidget, self).remove_layers_group()

        if self._active_image_group:
            self.image_groups_tw.clearSelection()
            self._active_image_group.setSelected(True)

    def remove_group(self):
        super(ImageGroupsManagerWidget, self).remove_group()

        self.image_groups_tw.clearSelection()
        self.groups_root.setSelected(True)


class LabelsManagerWidget(LabelsManager, QWidget):
    def __init__(self):
        super().__init__()

        self.labels_table_tw = QTreeWidget()
        self.labels_table_tw.setColumnCount(4)
        self.labels_table_tw.setHeaderLabels([
            "Acquisition value",
            "Sampling center",
            "Sampling top-left",
            "Sampling bottom-right"
        ])
        # self.labels_table_tw.itemSelectionChanged.connect(self.focus_region)
        self.labels_table_tw.itemPressed.connect(self.focus_region)

        self.labels_table_tw.addTopLevelItem(self.labels_group_root)
        self.labels_group_root.setExpanded(True)

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

        self.remove_labels_btn = QPushButton("Remove labels")
        self.remove_labels_btn.setEnabled(False)
        self.remove_labels_btn.clicked.connect(self.remove_labels)

        self.remove_labels_group_btn = QPushButton("Remove labels group")
        self.remove_labels_group_btn.setEnabled(False)
        self.remove_labels_group_btn.clicked.connect(self.remove_labels_group)

        manager_lyt = QGridLayout()
        manager_lyt.addWidget(self.labels_table_tw, 0, 0, 2, 4)
        manager_lyt.addWidget(self.prev_img_btn, 1, 0)
        manager_lyt.addWidget(self.prev_patch_btn, 1, 1)
        manager_lyt.addWidget(self.next_patch_btn, 1, 2)
        manager_lyt.addWidget(self.next_img_btn, 1, 3)
        manager_lyt.addWidget(self.edit_labels_btn, 2, 0, 1, 2)
        manager_lyt.addWidget(self.commit_btn, 2, 2, 1, 2)
        manager_lyt.addWidget(self.remove_labels_btn, 3, 0, 1, 2)
        manager_lyt.addWidget(self.remove_labels_group_btn, 3, 2, 1, 2)

        self.setLayout(manager_lyt)

    def focus_region(self, label: Optional[QTreeWidgetItem] = None,
                     edit_focused_label: bool = False):
        if not label:
            label = self.labels_table_tw.selectedItems()

        super(LabelsManagerWidget, self).focus_region(label,
                                                      edit_focused_label)

        self.edit_labels_btn.setEnabled(self._active_label is not None
                                        and not edit_focused_label)
        self.remove_labels_btn.setEnabled(self._active_label is not None)
        self.remove_labels_group_btn.setEnabled(
            self._active_label_group is not None
        )

        self.prev_img_btn.setEnabled(self._active_label is not None)
        self.prev_patch_btn.setEnabled(self._active_label is not None)
        self.next_patch_btn.setEnabled(self._active_label is not None)
        self.next_img_btn.setEnabled(self._active_label is not None)

    def edit_labels(self):
        editing = super(LabelsManagerWidget, self).edit_labels()
        self.commit_btn.setEnabled(editing)
        self.edit_labels_btn.setEnabled(not editing)

    def commit(self):
        super(LabelsManagerWidget, self).commit()
        self.commit_btn.setEnabled(False)
        self.edit_labels_btn.setEnabled(self._active_label is not None)

        self.prev_img_btn.setEnabled(self._active_label is not None)
        self.prev_patch_btn.setEnabled(self._active_label is not None)
        self.next_patch_btn.setEnabled(self._active_label is not None)
        self.next_img_btn.setEnabled(self._active_label is not None)

        self.remove_labels_btn.setEnabled(self._active_label is not None)
        self.remove_labels_group_btn.setEnabled(
            self._active_label_group is not None
        )


class TunableMethodWidget(TunableMethod, QWidget):
    def __init__(self):
        super().__init__()


class AcquisitionFunctionWidget(AcquisitionFunction, QWidget):
    def __init__(self, image_groups_manager: ImageGroupsManagerWidget,
                 labels_manager: LabelsManagerWidget,
                 tunable_segmentation_method: TunableMethodWidget):

        super().__init__(image_groups_manager, labels_manager,
                         tunable_segmentation_method)

        self.patch_sizes_mspn = MultiSpinBox()

        patch_sizes_scr = QScrollArea()
        patch_sizes_scr.setWidget(self.patch_sizes_mspn)
        patch_sizes_scr.setWidgetResizable(True)
        patch_sizes_scr.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        patch_sizes_lyt = QVBoxLayout()
        patch_sizes_lyt.addWidget(QLabel("Patch size:"))
        patch_sizes_lyt.addWidget(patch_sizes_scr)

        self.patch_sizes_widget = QWidget()
        self.patch_sizes_widget.setLayout(patch_sizes_lyt)

        patch_sizes_chk = QCheckBox("Edit patch sizes")
        patch_sizes_chk.setChecked(False)
        patch_sizes_chk.toggled.connect(self._show_patch_sizes)

        spatial_input_axes = self.input_axes
        if "C" in spatial_input_axes:
            spatial_input_axes = list(spatial_input_axes)
            spatial_input_axes.remove("C")
            spatial_input_axes = "".join(spatial_input_axes)

        self.patch_sizes_mspn.axes = spatial_input_axes
        self.patch_sizes_mspn.sizesChanged.connect(self._set_patch_size)

        self.max_samples_spn = QSpinBox(minimum=1, maximum=10000,
                                        value=self._max_samples,
                                        singleStep=10)
        self.max_samples_spn.valueChanged.connect(self._set_max_samples)

        self.MC_repetitions_spn = QSpinBox(minimum=2, maximum=100,
                                           value=self._MC_repetitions,
                                           singleStep=10)
        self.MC_repetitions_spn.valueChanged.connect(self._set_MC_repetitions)

        self.input_axes_le = QLineEdit()
        self.input_axes_le.textChanged.connect(self._set_input_axes)
        self.input_axes_le.setText(self.input_axes)

        self.model_axes_le = QLineEdit()
        self.model_axes_le.textChanged.connect(self._set_model_axes)
        self.model_axes_le.setText(self.model_axes)

        self.execute_selected_btn = QPushButton("Run on selected image groups")
        self.execute_selected_btn.clicked.connect(
            partial(self.compute_acquisition_layers, run_all=False)
        )

        self.execute_all_btn = QPushButton("Run on all image groups")
        self.execute_all_btn.clicked.connect(
            partial(self.compute_acquisition_layers, run_all=True)
        )

        self.image_pb = QProgressBar()
        self.patch_pb = QProgressBar()

        self.finetuning_btn = QPushButton("Fine tune model")
        self.finetuning_btn.clicked.connect(self.fine_tune)

        acquisition_lyt = QGridLayout()
        acquisition_lyt.addWidget(patch_sizes_chk, 0, 0)
        acquisition_lyt.addWidget(self.patch_sizes_widget, 1, 0, 1, 3)
        acquisition_lyt.addWidget(QLabel("Maximum samples:"), 2, 0)
        acquisition_lyt.addWidget(self.max_samples_spn, 2, 1)
        acquisition_lyt.addWidget(QLabel("Monte Carlo repetitions"), 3, 0)
        acquisition_lyt.addWidget(self.MC_repetitions_spn, 3, 1)
        acquisition_lyt.addWidget(QLabel("Input axes"), 4, 0)
        acquisition_lyt.addWidget(self.input_axes_le, 4, 1)
        acquisition_lyt.addWidget(QLabel("Model axes"), 4, 2)
        acquisition_lyt.addWidget(self.model_axes_le, 4, 3)
        acquisition_lyt.addWidget(self.tunable_segmentation_method, 5, 0, 1, 4)
        acquisition_lyt.addWidget(self.execute_selected_btn, 6, 0)
        acquisition_lyt.addWidget(self.execute_all_btn, 6, 1)
        acquisition_lyt.addWidget(self.finetuning_btn, 7, 1)
        acquisition_lyt.addWidget(QLabel("Image queue:"), 8, 0, 1, 1)
        acquisition_lyt.addWidget(self.image_pb, 8, 1, 1, 3)
        acquisition_lyt.addWidget(QLabel("Patch queue:"), 9, 0, 1, 1)
        acquisition_lyt.addWidget(self.patch_pb, 9, 1, 1, 3)

        self.setLayout(acquisition_lyt)
        self.patch_sizes_widget.setVisible(False)

    def _show_patch_sizes(self, show: bool):
        self.patch_sizes_widget.setVisible(show)

    def _reset_image_progressbar(self, num_images: int):
        self.image_pb.setRange(0, num_images)
        self.image_pb.reset()

    def _update_image_progressbar(self, curr_image_index: int):
        self.image_pb.setValue(curr_image_index)

    def _reset_patch_progressbar(self):
        self.patch_pb.setRange(0, self._max_samples)
        self.patch_pb.reset()

    def _update_patch_progressbar(self, curr_patch_index: int):
        self.patch_pb.setValue(curr_patch_index)

    def _set_patch_size(self, patch_sizes: dict):
        self._patch_sizes = patch_sizes

    def _set_MC_repetitions(self):
        self._MC_repetitions = self.MC_repetitions_spn.value()

    def _set_max_samples(self):
        self._max_samples = self.max_samples_spn.value()

    def _set_input_axes(self):
        self.input_axes = self.input_axes_le.text()
        self.patch_sizes_mspn.axes = self.input_axes

    def _set_model_axes(self):
        self.model_axes = self.model_axes_le.text()
