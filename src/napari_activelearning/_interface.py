from typing import Optional, Union, Iterable
from pathlib import Path
from qtpy.QtGui import QIntValidator
from qtpy.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
                            QGridLayout,
                            QLineEdit,
                            QComboBox,
                            QLabel,
                            QFileDialog,
                            QSpinBox,
                            QDoubleSpinBox,
                            QCheckBox,
                            QProgressBar,
                            QTreeWidget,
                            QTreeWidgetItem,
                            QAbstractItemView)

from functools import partial
import napari

from ._acquisition import AcquisitionFunction, TunableMethod
from ._layers import (ImageGroupEditor, ImageGroupsManager, LayerScaleEditor,
                      MaskGenerator,
                      ImageGroup,
                      ImageGroupRoot)
from ._labels import LayersGroup, LayerChannel, LabelsManager


class ImageGroupEditorWidget(ImageGroupEditor, QWidget):
    def __init__(self):
        super().__init__()

        layout = QGridLayout()
        self.group_name_le = QLineEdit("None selected")
        self.group_name_le.setEnabled(False)
        self.group_name_le.returnPressed.connect(self.update_group_name)
        layout.addWidget(QLabel("Group name:"), 0, 0)
        layout.addWidget(self.group_name_le, 0, 1)

        self.layers_group_name_cmb = QComboBox()
        self.layers_group_name_cmb.setEditable(True)
        self.layers_group_name_cmb.lineEdit().returnPressed.connect(
            self.update_layers_group_name
        )
        self.layers_group_name_cmb.setEnabled(False)
        layout.addWidget(QLabel("Channels group name:"), 0, 2)
        layout.addWidget(self.layers_group_name_cmb, 0, 3)

        self.display_name_lbl = QLabel("None selected")
        layout.addWidget(QLabel("Channel name:"), 1, 0)
        layout.addWidget(self.display_name_lbl, 1, 1)

        self.edit_channel_spn = QSpinBox(minimum=0, maximum=0)
        self.edit_channel_spn.setEnabled(False)
        self.edit_channel_spn.editingFinished.connect(self.update_channels)
        self.edit_channel_spn.valueChanged.connect(self.update_channels)
        layout.addWidget(QLabel("Channel:"), 1, 2)
        layout.addWidget(self.edit_channel_spn, 1, 3)

        self.edit_axes_le = QLineEdit("None selected")
        self.edit_axes_le.setEnabled(False)
        self.edit_axes_le.returnPressed.connect(self.update_source_axes)
        layout.addWidget(QLabel("Axes order:"), 2, 0)
        layout.addWidget(self.edit_axes_le, 2, 1)

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
        layout.addWidget(QLabel("Output directory:"), 3, 0)
        layout.addWidget(self.output_dir_le, 3, 1, 1, 3)
        layout.addWidget(self.output_dir_btn, 3, 3)

        self.use_as_input_chk = QCheckBox("Use as input")
        self.use_as_input_chk.setEnabled(False)
        self.use_as_input_chk.toggled.connect(
            self.update_use_as_input
        )
        layout.addWidget(self.use_as_input_chk, 4, 0)

        self.use_as_sampling_chk = QCheckBox("Use as sampling mask")
        self.use_as_sampling_chk.setEnabled(False)
        self.use_as_sampling_chk.toggled.connect(
            self.update_use_as_sampling
        )
        layout.addWidget(self.use_as_sampling_chk, 4, 1)

        self.setLayout(layout)

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

    def _clear_layer_channel(self):
        self.display_name_lbl.setText("None selected")
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

        display_source_axes = list(self._edit_axes.lower())
        if "c" in display_source_axes:
            display_source_axes.remove("c")
        display_source_axes = tuple(display_source_axes)

        viewer = napari.current_viewer()
        if display_source_axes != viewer.dims.axis_labels:
            viewer.dims.axis_labels = display_source_axes

    def update_layers_group_name(self):
        if super().update_layers_group_name(
          self.layers_group_name_cmb.lineEdit().text()):
            self._clear_layer_channel()

    def update_use_as_input(self):
        super().update_use_as_input(self.use_as_input_chk.isChecked())

    def update_use_as_sampling(self):
        super().update_use_as_sampling(self.use_as_sampling_chk.isChecked())

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


class LayerScaleEditorWidget(LayerScaleEditor, QWidget):
    def __init__(self):
        super().__init__()

        self.edit_scale_lyt = QVBoxLayout()
        self.edit_scale_lyt.addWidget(QLabel("Channel(s) scale(s):"))
        self._curr_scale_spn_list = []

        self.setLayout(self.edit_scale_lyt)

    def _clear_layer_channel(self):
        while self._curr_scale_spn_list:
            edit_scale_spn = self._curr_scale_spn_list.pop()
            self.edit_scale_lyt.removeWidget(edit_scale_spn)

        self._curr_scale_spn_list.clear()

    def _fill_layer(self):
        self._clear_layer_channel()

        if self._active_layer_channel:
            scales = self._active_layer_channel.scale

            for s in scales:
                edit_scale_spn = QDoubleSpinBox(minimum=1e-12, maximum=1e12,
                                                singleStep=1e-7,
                                                decimals=7)
                edit_scale_spn.setValue(s)
                edit_scale_spn.lineEdit().returnPressed.connect(
                    self.update_scale
                )
                self.edit_scale_lyt.addWidget(edit_scale_spn)
                self._curr_scale_spn_list.append(edit_scale_spn)

    def update_scale(self):
        new_scale = [
            edit_scale_spn.value()
            for edit_scale_spn in self._curr_scale_spn_list
        ]
        super().update_scale(new_scale)

    @property
    def active_layer_channel(self):
        return super().active_layer_channel

    @active_layer_channel.setter
    def active_layer_channel(self,
                             active_layer_channel: Union[LayerChannel, None]):
        super(LayerScaleEditor, type(self)).active_layer_channel\
                                            .fset(self, active_layer_channel)
        self._fill_layer()


class MaskGeneratorWidget(MaskGenerator, QWidget):
    def __init__(self):
        super().__init__()

        self.patch_size_lbl = QLabel("Patch size")

        self.patch_size_le = QLineEdit("128")
        self.patch_size_le.setValidator(QIntValidator(0, 2**16))
        self.patch_size_le.returnPressed.connect(self._set_patch_size)

        self.patch_size_spn = QSpinBox(minimum=0, maximum=16, singleStep=1)
        self.patch_size_spn.setValue(7)
        self.patch_size_spn.lineEdit().hide()
        self.patch_size_spn.setEnabled(False)
        self.patch_size_spn.valueChanged.connect(self._modify_patch_size)

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

    def _modify_patch_size(self, scale: int):
        self.patch_size_le.setText(str(2 ** scale))
        self._set_patch_size()

    def _set_patch_size(self):
        super().set_patch_size(int(self.patch_size_le.text()))

    @property
    def active_image_group(self):
        return super().active_image_group

    @active_image_group.setter
    def active_image_group(self, active_image_group: ImageGroup):
        super(MaskGeneratorWidget, type(self)).active_image_group\
                                              .fset(self, active_image_group)

        self.generate_mask_btn.setEnabled(
            self._active_image_group is not None
        )
        self.patch_size_le.setEnabled(
            self._active_image_group is not None
        )
        self.patch_size_spn.setEnabled(
            self._active_image_group is not None
        )


class ImageGroupsManagerWidget(ImageGroupsManager, QWidget):
    def __init__(self, default_axis_labels: str = "TZYX"):
        super().__init__(default_axis_labels)

        self.image_groups_editor = ImageGroupEditorWidget()
        self.layer_scale_editor = LayerScaleEditorWidget()
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
            self._focus_active_element
        )
        self.image_groups_tw.itemSelectionChanged.connect(
            self._get_active_item
        )
        self.image_groups_tw.setExpandsOnDoubleClick(False)
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
        self.save_layers_group_btn.clicked.connect(self._save_layers_group)
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

        self.group_lyt = QVBoxLayout()
        self.group_lyt.addLayout(self.group_buttons_lyt)
        self.group_lyt.addLayout(self.layers_group_buttons_lyt)
        self.group_lyt.addLayout(self.layer_buttons_lyt)
        self.group_lyt.addWidget(self.show_editor_chk)
        self.group_lyt.addWidget(self.image_groups_editor)
        self.group_lyt.addWidget(self.layer_scale_editor)
        self.group_lyt.addWidget(self.mask_generator)
        self.group_lyt.addWidget(self.image_groups_tw)

        self.setLayout(self.group_lyt)

        self.image_groups_editor.setVisible(False)
        self.layer_scale_editor.setVisible(False)
        self.mask_generator.setVisible(False)

    def _show_editor(self, show: bool = False):
        self.image_groups_editor.setVisible(show)
        self.layer_scale_editor.setVisible(show)
        self.mask_generator.setVisible(show)

    def _save_layers_group(self):
        self.save_layers_group()
        self.save_layers_group_btn.setEnabled(False)

    def _get_active_item(self, item: Optional[
                             Union[QTreeWidgetItem, Iterable[QTreeWidgetItem]]
                             ] = None):
        item_list = list(filter(
            lambda item: isinstance(item, (ImageGroupRoot,
                                           ImageGroup,
                                           LayersGroup,
                                           LayerChannel)),
            self.image_groups_tw.selectedItems()
        ))

        super(ImageGroupsManagerWidget, self)._get_active_item(item_list)

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
            and not isinstance(self._active_layers_group.source_data,
                               (str, Path))
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
        self.labels_table_tw.itemSelectionChanged.connect(self.focus_region)

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

    def focus_region(self, label: Optional[QTreeWidgetItem] = None,
                     edit_focused_label: bool = False):
        if not label:
            label = self.labels_table_tw.selectedItems()

        self.edit_labels_btn.setEnabled(
            super(LabelsManagerWidget, self).focus_region(label,
                                                          edit_focused_label)
        )

    def edit_labels(self):
        editing = super(LabelsManagerWidget, self).edit_labels()
        self.commit_btn.setEnabled(editing)
        self.edit_labels_btn.setEnabled(editing)

    def commit(self):
        super(LabelsManagerWidget, self).commit()
        self.commit_btn.setEnabled(False)
        self.edit_labels_btn.setEnabled(True)


class TunableMethodWidget(QWidget, TunableMethod):
    def __init__(self):
        super().__init__()


class AcquisitionFunctionWidget(AcquisitionFunction, QWidget):
    def __init__(self, image_groups_manager: ImageGroupsManagerWidget,
                 labels_manager: LabelsManagerWidget,
                 tunable_segmentation_method: TunableMethodWidget):

        super().__init__(image_groups_manager, labels_manager,
                         tunable_segmentation_method)

        self.patch_size_lbl = QLabel("Patch size:")
        self.patch_size_spn = QSpinBox(minimum=128, maximum=1024,
                                       singleStep=128)
        self.patch_size_spn.valueChanged.connect(self._set_patch_size)

        self.patch_size_lyt = QHBoxLayout()
        self.patch_size_lyt.addWidget(self.patch_size_lbl)
        self.patch_size_lyt.addWidget(self.patch_size_spn)

        self.max_samples_lbl = QLabel("Maximum samples:")
        self.max_samples_spn = QSpinBox(minimum=1, maximum=10000, value=100,
                                        singleStep=10)
        self.max_samples_spn.valueChanged.connect(self._set_max_samples)

        self.max_samples_lyt = QHBoxLayout()
        self.max_samples_lyt.addWidget(self.max_samples_lbl)
        self.max_samples_lyt.addWidget(self.max_samples_spn)

        self.MC_repetitions_lbl = QLabel("Monte Carlo repetitions")
        self.MC_repetitions_spn = QSpinBox(minimum=1, maximum=100, value=30,
                                           singleStep=10)
        self.MC_repetitions_spn.valueChanged.connect(self._set_MC_repetitions)

        self.MC_repetitions_lyt = QHBoxLayout()
        self.MC_repetitions_lyt.addWidget(self.MC_repetitions_lbl)
        self.MC_repetitions_lyt.addWidget(self.MC_repetitions_spn)

        self.execute_selected_btn = QPushButton("Run on selected image groups")
        self.execute_selected_btn.clicked.connect(
            partial(self.compute_acquisition_layers, run_all=False)
        )

        self.execute_all_btn = QPushButton("Run on all image groups")
        self.execute_all_btn.clicked.connect(
            partial(self.compute_acquisition_layers, run_all=True)
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

        self.finetuning_btn = QPushButton("Fine tune model")
        self.finetuning_btn.clicked.connect(self.fine_tune)

        self.execute_lyt = QHBoxLayout()
        self.execute_lyt.addWidget(self.execute_selected_btn)
        self.execute_lyt.addWidget(self.execute_all_btn)

        self.acquisition_lyt = QVBoxLayout()
        self.acquisition_lyt.addLayout(self.patch_size_lyt)
        self.acquisition_lyt.addLayout(self.max_samples_lyt)
        self.acquisition_lyt.addLayout(self.MC_repetitions_lyt)
        self.acquisition_lyt.addWidget(self.tunable_segmentation_method)
        self.acquisition_lyt.addLayout(self.execute_lyt)
        self.acquisition_lyt.addLayout(self.image_lyt)
        self.acquisition_lyt.addLayout(self.patch_lyt)
        self.acquisition_lyt.addWidget(self.finetuning_btn)

        self.setLayout(self.acquisition_lyt)

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

    def _set_patch_size(self):
        self._patch_size = self.patch_size_spn.value()

    def _set_MC_repetitions(self):
        self._MC_repetitions = self.MC_repetitions_spn.value()

    def _set_max_samples(self):
        self._max_samples = self.max_samples_spn.value()
