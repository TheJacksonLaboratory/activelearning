from qtpy.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
                            QLineEdit,
                            QLabel,
                            QFileDialog,
                            QSpinBox,
                            QProgressBar)

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


def  compare_layers(layer, ref_layer=None, compare_type=True,
                   compare_shape=True):
    ref_layer_base_name = " ".join(ref_layer.name.split(" ")[:-1])

    if ref_layer_base_name not in layer.name:
        return False

    if isinstance(layer, type(ref_layer)):
        if not layer._source.path == ref_layer._source.path:
            return False

    elif compare_type:
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
    out_grp = zarr.open(output_filename, mode="a")

    if isinstance(chunk_size, int):
        chunk_size = [chunk_size] * len(shape)

    chunks_size_axes = list(map(min, shape, chunk_size))

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


def downsample_image(filename, source_axes, data_group, scale=4, num_scales=5,
                     reference_source_axes=None,
                     reference_scale=None,
                     reference_units=None):
    source_arr = da.from_zarr(filename, component=data_group)

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

        target_arr.to_zarr(filename,
                           component=root_group % s,
                           compressor=zarr.Blosc(clevel=9),
                           write_empty_chunks=False,
                           overwrite=True)

        source_arr = da.from_zarr(filename,
                                  component=root_group % s)

        datasets.append({
            "coordinateTransformations": [
                {"type": "scale",
                 "scale": [4.0 ** s * reference_scale_axes.get(ax, 1.0)
                           if ax in "ZYX" else 1.0
                           for ax in source_axes]
                }],
            "path": str(s)
        })

    root = zarr.open(Path(filename) / data_group, mode="a")
    write_multiscales_metadata(root, datasets, axes=list(source_axes.lower()))


class MaskGenerator(QWidget):
    def __init__(self, viewer):
        super().__init__()

        self.viewer = viewer

        self.patch_size_lbl = QLabel("Patch size")

        self.patch_size_spn = QSpinBox(minimum=128, maximum=4096,
                                       singleStep=16)

        self.patch_size_lyt = QHBoxLayout()
        self.patch_size_lyt.addWidget(self.patch_size_lbl)
        self.patch_size_lyt.addWidget(self.patch_size_spn)

        self.generate_mask_btn = QPushButton("Generate mask")
        self.generate_mask_btn.clicked.connect(self.generate_masks)

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.patch_size_lyt)
        self.layout.addWidget(self.generate_mask_btn)

        self.setLayout(self.layout)

    def _generate_mask_layer(self, image_layer):
        patch_size = self.patch_size_spn.value()

        if image_layer.metadata["splitted_channels"]:
            displayed_source_axes = [
                ax
                for ax in image_layer.metadata["source_axes"]
                if ax != "C"
            ]

            displayed_roi = tuple(
                ax_roi
                for ax, ax_roi in zip(image_layer.metadata["source_axes"],
                                      image_layer.metadata["roi"][0])
                if ax != "C"
            )

        else:
            displayed_source_axes = image_layer.metadata["source_axes"]
            displayed_roi = image_layer.metadata["roi"][0]

        im_shape = image_layer.data.shape
        im_scale = image_layer.scale
        im_trans = image_layer.translate

        mask_axes = "".join([
            ax
            for ax, ax_roi in zip(displayed_source_axes, displayed_roi)
            if ax in "ZYX" and ax_roi.stop is None
        ])

        mask_shape = [int(np.ceil(s / patch_size))
                      for s, ax in zip(im_shape, displayed_source_axes)
                      if ax in mask_axes]

        mask_scale = tuple(
            (ax_scl * patch_size) if ax_s // patch_size > 1 else ax_scl
            for ax, ax_s, ax_scl in zip(displayed_source_axes, im_shape,
                                        im_scale)
            if ax in mask_axes
        )

        mask_translate = tuple(
            ax_trans
            + ((ax_scl * (patch_size - 1) / 2)
               if ax_s // patch_size > 1 else 0)
            for ax, ax_s, ax_scl, ax_trans in zip(displayed_source_axes,
                                                  im_shape,
                                                  im_scale,
                                                  im_trans)
            if ax in mask_axes
        )

        self.viewer.add_labels(data=np.zeros(mask_shape, dtype=np.uint8),
                               scale=mask_scale,
                               translate=mask_translate,
                               name=image_layer.name + " sample-mask",
                               metadata={"source_axes": mask_axes})

    def generate_masks(self):
        selected_image_layers = list(filter(
            lambda curr_layer: isinstance(curr_layer, Image),
            self.viewer.layers.selection
        ))

        # Remove sibling layers
        layers_to_mask = []
        while selected_image_layers:
            reference_layer = selected_image_layers.pop()

            sibling_layers = list(filter(
                partial(compare_layers, ref_layer=reference_layer,
                        compare_type=True,
                        compare_shape=True),
                selected_image_layers
            ))

            for curr_layer in sibling_layers:
                if curr_layer in selected_image_layers:
                    selected_image_layers.remove(curr_layer)

            layers_to_mask.append(reference_layer)

        for curr_layer in layers_to_mask:
            self._generate_mask_layer(curr_layer)


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
    def __init__(self, viewer: napari.Viewer, labels_manager: LabelsManager):
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

        self.output_dir_lbl = QLabel("Output directory:")
        self.output_dir_le = QLineEdit(".")
        self.output_dir_dlg = QFileDialog(fileMode=QFileDialog.Directory)
        self.output_dir_btn = QPushButton("...")
        self.output_dir_lyt = QHBoxLayout()
        self.output_dir_lyt.addWidget(self.output_dir_lbl)
        self.output_dir_lyt.addWidget(self.output_dir_le)
        self.output_dir_lyt.addWidget(self.output_dir_btn)

        self.output_dir_btn.clicked.connect(self.output_dir_dlg.show)
        self.output_dir_dlg.directoryEntered.connect(self._update_output_dir)

        self.execute_btn = QPushButton("Run on selected layers")
        self.execute_btn.clicked.connect(self.compute_acquisition_fun)

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

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.patch_size_lyt)
        self.layout.addLayout(self.max_samples_lyt)
        self.layout.addLayout(self.MC_repetitions_lyt)
        self.layout.addLayout(self.output_dir_lyt)
        self.layout.addWidget(self.execute_btn)
        self.layout.addLayout(self.image_lyt)
        self.layout.addLayout(self.patch_lyt)
        self.setLayout(self.layout)

        self.labels_manager = labels_manager

    def _update_output_dir(self, output_dir):
        self.output_dir_le.setText(output_dir)

    def _compute_acquisition_fun_layer(self, image_layer, mask_layer=None):
        patch_size = self.patch_size_spn.value()
        MC_repetitions = self.MC_repetitions_spn.value()
        max_samples = self.max_samples_spn.value()
        output_dir = Path(self.output_dir_le.text())

        output_name = image_layer.name

        # Remove any invalid character from the name
        for chr in [" ", ".", "/", "\\"]:
            output_name = "-".join(output_name.split(chr))

        if image_layer.metadata["splitted_channels"]:
            displayed_source_axes = [
                ax
                for ax in image_layer.metadata["source_axes"]
                if ax != "C"
            ]

            displayed_roi = [
                ax_roi
                for ax, ax_roi in zip(image_layer.metadata["source_axes"],
                                      image_layer.metadata["roi"][0])
                if ax != "C"
            ]

        else:
            displayed_source_axes = image_layer.metadata["source_axes"]
            displayed_roi = image_layer.metadata["roi"][0]

        acquisition_fun_shape = [
            ax_s
            for ax, ax_s in zip(displayed_source_axes, image_layer.data.shape)
            if ax in "ZYX" and ax_s > 1
        ]

        acquisition_fun_scale = [
            ax_scl
            for ax, ax_s, ax_scl in zip(displayed_source_axes,
                                        image_layer.data.shape,
                                        image_layer.scale)
            if ax in "ZYX" and ax_s > 1
        ]

        acquisition_fun_chunk_size = [max(patch_size, 4096)]
        acquisition_fun_chunk_size *= len(acquisition_fun_shape)

        if image_layer.metadata["filenames"]:
            source_data = image_layer.metadata["filenames"]

        elif image_layer.metadata["splitted_channels"]:
            splitted_layers = filter(
                partial(compare_layers, ref_layer=image_layer,
                        compare_type=True,
                        compare_shape=True),
                viewer.layers
            )

            splitted_layers_data = list(map(lambda curr_layer:
                                            curr_layer.data,
                                            splitted_layers))

            source_data = np.stack(
                splitted_layers_data,
                axis=image_layer.metadata["source_axes"].index("C")
            )

        else:
            source_data = image_layer.data

        dataset_metadata = {
            "images": zds.ImagesDatasetSpecs(
                filenames=source_data,
                data_group=image_layer.metadata["data_group"],
                source_axes=image_layer.metadata["source_axes"],
                axes=image_layer.metadata["axes"],
                roi=image_layer.metadata["roi"],
            ),
        }

        output_name = image_layer.name

        # Remove any invalid character from the name
        for chr in [" ", ".", "/", "\\"]:
            output_name = "-".join(output_name.split(chr))

        output_filename = output_dir / (output_name + ".zarr")
        labels_metadata = []

        if mask_layer:
            mask_axes = "".join([
                ax
                for ax, ax_roi in zip(displayed_source_axes, displayed_roi)
                if ax in "ZYX" and ax_roi.stop is None
            ])

            if mask_layer._source.path is None:
                save_zarr(output_filename, data=mask_layer.data,
                          shape=mask_layer.data.shape,
                          chunk_size=max(1, int(patch_size ** 0.5)),
                          group_name="labels/sampling_mask",
                          dtype=bool)

                mask_source_data = str(output_filename)
                mask_data_group = "labels/sampling_mask"

                labels_metadata.append({
                    "name": "labels/sampling_mask",
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
                })

            else:
                mask_source_data = mask_layer.data
                mask_data_group = None

            dataset_metadata["masks"] = zds.MasksDatasetSpecs(
                filenames=mask_source_data,
                data_group=mask_data_group,
                source_axes=mask_axes,
                axes=mask_axes
            )

        save_zarr(output_filename, data=None,
                  shape=acquisition_fun_shape,
                  chunk_size=acquisition_fun_chunk_size,
                  group_name="labels/acquisition_fun/0",
                  dtype=np.float32)

        save_zarr(output_filename, data=None, shape=acquisition_fun_shape,
                  chunk_size=acquisition_fun_chunk_size,
                  group_name="labels/segmentation/0",
                  dtype=np.int32)

        acquisition_fun = zarr.open(str(output_filename)
                                    + "/labels/acquisition_fun/0",
                                    mode="a")
        segmentation_out = zarr.open(str(output_filename)
                                     + "/labels/segmentation/0",
                                     mode="a")

        dl = datautils.get_dataloader(dataset_metadata, patch_size=patch_size,
                                      shuffle=True)

        model_dropout = cellpose_model_init(use_dropout=True)
        model = cellpose_model_init(use_dropout=False)

        acquisition_fun_max = 0
        segmentation_max = 0
        n_samples = 0
        img_sampling_positions = []

        self.patch_pb.setRange(0, max_samples)
        self.patch_pb.reset()
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

            self.patch_pb.setValue(n_samples)
            if n_samples >= max_samples:
                break

        self.patch_pb.setValue(max_samples)
        layer_name = image_layer.name + " segmentation output"
        self.labels_manager.sampling_positions[layer_name] =\
            img_sampling_positions

        # Downsample the acquisition function
        downsample_image(
            output_filename,
            source_axes="YX",
            data_group="labels/acquisition_fun/0",
            scale=4,
            num_scales=5,
            reference_source_axes=displayed_source_axes,
            reference_scale=image_layer.scale
        )

        viewer.open(
            str(output_filename) + "/labels/acquisition_fun",
            layer_type="image",
            name=image_layer.name + " acquisition function",
            multiscale=True,
            opacity=0.8,
            scale=acquisition_fun_scale,
            blending="translucent_no_depth",
            colormap="magma",
            contrast_limits=(0, acquisition_fun_max)
        )

        # Downsample the segmentation output
        downsample_image(
            output_filename,
            source_axes="YX",
            data_group="labels/segmentation/0",
            scale=4,
            num_scales=5,
            reference_source_axes=displayed_source_axes,
            reference_scale=image_layer.scale,
        )

        viewer.open(
            str(output_filename) + "/labels/segmentation",
            layer_type="labels",
            name=image_layer.name + " segmentation output",
            multiscale=True,
            scale=acquisition_fun_scale,
            opacity=0.8
        )

        z_grp = zarr.open(output_filename, mode="a")
        for lab_meta in labels_metadata:
            write_label_metadata(z_grp, **lab_meta)

    def compute_acquisition_fun(self):
        selected_image_layers = list(filter(
            lambda curr_layer: isinstance(curr_layer, Image),
            self.viewer.layers.selection
        ))

        layers_masks = []

        while selected_image_layers:
            reference_layer = selected_image_layers.pop()

            sibling_layers = list(filter(
                partial(compare_layers, ref_layer=reference_layer,
                        compare_type=False,
                        compare_shape=True),
                viewer.layers
            ))

            for curr_layer in sibling_layers:
                if curr_layer in selected_image_layers:
                    selected_image_layers.remove(curr_layer)

            mask_layers = list(filter(
                lambda curr_layer: isinstance(curr_layer, Labels),
                sibling_layers
            ))

            image_layers = list(filter(
                lambda curr_layer: isinstance(curr_layer, Image),
                sibling_layers
            ))

            if len(mask_layers):
                mask_layers = mask_layers[0]
            else:
                mask_layers = None

            layers_masks.append((image_layers[0], mask_layers))

        self.image_pb.setRange(0, len(layers_masks))
        self.image_pb.reset()
        for n, (image_layer, mask_layer) in enumerate(layers_masks):
            self._compute_acquisition_fun_layer(image_layer, mask_layer)
            self.image_pb.setValue(n + 1)


class MetadataManager(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        self.source_axes_lbl = QLabel("Axes order:")

        self.source_axes_le = QLineEdit("None selected")
        self.source_axes_le.returnPressed.connect(self.update_metadata)

        self.update_btn = QPushButton("udpdate")
        self.update_btn.clicked.connect(self.update_metadata)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.source_axes_lbl)
        self.layout.addWidget(self.source_axes_le)
        self.layout.addWidget(self.update_btn)

        self.setLayout(self.layout)

        self.viewer.layers.selection.events.connect(self._show_source_axes)

    def _show_source_axes(self, event):
        if not len(event._sources) or not all(event._sources):
            self.source_axes_le.setText("None selected")
            return

        selected_layer = next(iter(event._sources[0]))
        source_axes = selected_layer.metadata.get("source_axes", "unset")

        self.source_axes_le.setText(source_axes)

    def _update_metadata_layer(self, layer):
        source_axes = self.source_axes_le.text()

        input_filename = layer._source.path
        data_group = ""

        if input_filename:
            input_filename = str(input_filename)
            data_group = "/".join(input_filename.split(".")[-1].split("/")[1:])

        if data_group:
            input_filename = input_filename[:-len(data_group) - 1]

        if isinstance(layer.data, MultiScaleData):
            data_group += "/0"

        if not data_group:
            data_group = None

        # Determine if the current layer was generated by splitting the
        # channels of an image.
        splitted_layers = list(filter(
            partial(compare_layers, ref_layer=layer, compare_type=True,
                    compare_shape=True),
            self.viewer.layers
        ))

        is_splitted = len(splitted_layers) > 1

        # Generate the current ROI for images with non-channel, non-spatial
        # axes.
        if is_splitted:
            displayed_source_axes = [ax for ax in source_axes if ax != "C"]
        else:
            displayed_source_axes = source_axes

        roi_start = [0] * len(displayed_source_axes)
        roi_length = [-1] * len(displayed_source_axes)

        for ord in viewer.dims.order[:-viewer.dims.ndisplay]:
            roi_start[ord] = int(viewer.cursor.position[ord])
            roi_length[ord] = 1

        if is_splitted and "C" in source_axes:
            roi_start.insert(source_axes.index("C"), 0)
            roi_length.insert(source_axes.index("C"), -1)

        axes = ["Y", "X", "C"]
        if "C" not in source_axes:
            axes.remove("C")
        axes = "".join(axes)

        roi = [
            tuple(slice(ax_start if ax_length > 0 else None,
                        (ax_start + ax_length) if ax_length > 0 else None)
                  for ax_start, ax_length in zip(roi_start, roi_length))
        ]

        # Apply to all layers added from this same input file that might have
        # been splitted into channels.
        for curr_layer in splitted_layers:
            curr_layer.metadata["splitted_channels"] = is_splitted
            curr_layer.metadata["filenames"] = input_filename
            curr_layer.metadata["data_group"] = data_group
            curr_layer.metadata["source_axes"] = source_axes
            curr_layer.metadata["axes"] = axes
            curr_layer.metadata["roi"] = roi

    def update_metadata(self):
        selected_image_layers = list(self.viewer.layers.selection)

        # Add sibling layers
        while selected_image_layers:
            reference_layer = selected_image_layers.pop()

            sibling_layers = list(filter(
                partial(compare_layers, ref_layer=reference_layer,
                        compare_type=True,
                        compare_shape=True),
                self.viewer.layers
            ))

            for curr_layer in sibling_layers:
                if curr_layer in selected_image_layers:
                    selected_image_layers.remove(curr_layer)

                self._update_metadata_layer(curr_layer)


if __name__ == "__main__":
    viewer = napari.Viewer()

    metadata_manager = MetadataManager(viewer)
    mask_generator = MaskGenerator(viewer)
    labels_manager_widget = LabelsManager(viewer)
    acquisition_function = AcquisitionFunction(viewer, labels_manager_widget)

    viewer.window.add_dock_widget(metadata_manager)
    viewer.window.add_dock_widget(mask_generator)
    viewer.window.add_dock_widget(acquisition_function)
    viewer.window.add_dock_widget(labels_manager_widget, area='right')

    napari.run()
