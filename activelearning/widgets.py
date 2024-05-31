from qtpy.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
                            QLineEdit,
                            QLabel,
                            QFileDialog,
                            QSpinBox)

from functools import partial
from pathlib import Path
import torch
import napari
from napari.layers import Image, Labels
from napari.layers._multiscale_data import MultiScaleData
import tensorstore as ts
import numpy as np
import zarr
from ome_zarr.writer import write_multiscales_metadata
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


def compare_layers(layer, ref_layer=None, compare_type=True,
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
        layer_axes = layer.metadata.get("source_axes", None)
        ref_layer_axes = layer.metadata.get("source_axes", None)

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

    # TODO: Write metadata so napari (or any ome-zarr reader) can open these files


def downsample_image(filename, source_axes, data_group=None, scale=4,
                     num_scales=5):
    if data_group is not None:
        root_group = "/".join(data_group.split("/")[:-1]) + "/%i"
    else:
        root_group = "%i"

    source_arr = da.from_zarr(filename, component=data_group)
    min_size = min(source_arr.shape[source_axes.index(ax)]
                   for ax in "YX" if ax in source_axes
                   )
    num_scales = min(num_scales, int(np.log(min_size) / np.log(scale)))

    downscale_selection = tuple(
        slice(None, None, scale) if ax in "ZYX" and ax_s > 1 else slice(None)
        for ax, ax_s in zip(source_axes, source_arr.shape)
    )

    for s in range(1, num_scales):
        target_arr = source_arr[downscale_selection]
        target_arr = target_arr.rechunk(
            tuple(
                tuple(chk // scale for chk in chk_ax)
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

    # TODO: Write mutliscales metadata so napari (or any ome-zarr reader) can open these files
    write_multiscales_metadata


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
        for layer in self.viewer.layers.selection:
            if not isinstance(layer, Image):
                continue

            self._generate_mask_layer(layer)


class AcquisitionFunction(QWidget):
    def __init__(self, viewer):
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

        self.output_dir_fe = FileEdit(mode="d", label="Ouptut directory")

        self.execute_btn = QPushButton("Execute")
        self.execute_btn.clicked.connect(self.compute_acquisition_fun)

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.patch_size_lyt)
        self.layout.addLayout(self.max_samples_lyt)
        self.layout.addLayout(self.MC_repetitions_lyt)
        self.layout.addLayout(self.output_dir_lyt)
        self.layout.addWidget(self.execute_btn)
        self.setLayout(self.layout)

        self.sampling_positions = {}

        self.viewer.layers.events.removing.connect(
            self._remove_sampling_positions
        )

    def _update_output_dir(self, output_dir):
        self.output_dir_le.setText(output_dir)

    def _remove_sampling_positions(self, event):
        for removed_layers in event._sources[0]:
            output_name = removed_layers.name

            # Remove any invalid character from the name
            for chr in [" ", ".", "/", "\\"]:
                output_name = "-".join(output_name.split(chr))

            if output_name in self.sampling_positions:
                self.sampling_positions.pop(output_name)

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
            s
            for s, ax in zip(image_layer.data.shape, displayed_source_axes)
            if ax in "ZYX" and s > 1
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

        output_filename = output_dir / Path(output_name + ".zarr")

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
                          group_name="sampling_mask",
                          dtype=bool)

                mask_source_data = str(output_filename)
                mask_data_group = "sampling_mask"

            else:
                mask_source_data = mask_layer.data
                mask_data_group = None

            dataset_metadata["masks"] = zds.MasksDatasetSpecs(
                filenames=mask_source_data,
                data_group=mask_data_group,
                source_axes=mask_axes,
                axes=mask_axes
            )

        save_zarr(output_filename, data=None, shape=acquisition_fun_shape,
                  chunk_size=acquisition_fun_chunk_size,
                  group_name="acquisition_fun/0",
                  dtype=np.float32)

        save_zarr(output_filename, data=None, shape=acquisition_fun_shape,
                  chunk_size=acquisition_fun_chunk_size,
                  group_name="segmentation/0",
                  dtype=np.int32)

        acquisition_fun = zarr.open(str(output_filename)
                                    + "/acquisition_fun/0",
                                    mode="a")
        segmentation_out = zarr.open(str(output_filename) + "/segmentation/0",
                                     mode="a")

        dl = datautils.get_dataloader(dataset_metadata, patch_size=patch_size,
                                      shuffle=True)

        model_dropout = cellpose_model_init(use_dropout=True)
        model = cellpose_model_init(use_dropout=False)

        acquisition_fun_max = 0
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

            seg_out = cellpose_segment(img[0].numpy(), model)
            segmentation_out[pos_u_lab] = seg_out

            img_sampling_positions.append(pos_u_lab)

            n_samples += 1

            if n_samples >= max_samples:
                break

        # Downsample the acquisition function
        downsample_image(output_filename, "YX", data_group="acquisition_fun/0",
                         scale=4,
                         num_scales=5)

        viewer.open(str(output_filename) + "/acquisition_fun",
                    name=image_layer.name + " acquisition function",
                    opacity=0.8,
                    blending="translucent_no_depth",
                    colormap="magma",
                    contrast_limits=(0, acquisition_fun_max))

        # Downsample the segmentation output
        downsample_image(output_filename, "YX", data_group="segmentation/0",
                         scale=4,
                         num_scales=5)

        viewer.open(str(output_filename) + "/segmentation",
                    name=image_layer.name + " segmentation output",
                    opacity=0.8)

        self.sampling_positions[image_layer.name + " acquisition function"] =\
            img_sampling_positions

    def compute_acquisition_fun(self):
        selected_image_layers = list(filter(
            lambda curr_layer: isinstance(curr_layer, Image),
            self.viewer.layers.selection
        ))

        layers_masks = []

        while selected_image_layers:
            layer = selected_image_layers.pop()

            sibling_layers = list(filter(
                partial(compare_layers, ref_layer=layer, compare_type=False,
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

        for image_layer, mask_layer in layers_masks:
            self._compute_acquisition_fun_layer(image_layer, mask_layer)


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
        for layer in self.viewer.layers.selection:
            self._update_metadata_layer(layer)


class LabelsManager(QWidget):
    def __init__(self, viewer: napari.Viewer,
                 acquisition_function: AcquisitionFunction) -> None:
        super().__init__()

        self.viewer = viewer
        self.acquisition_function = acquisition_function

        self._current_patch_position = 0
        self._current_image = 0

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

    def _previus_image(self):
        self.navigate(delta_image_index=-1)

    def _previus_patch(self):
        self.navigate(delta_patch_index=-1)

    def _next_image(self):
        self.navigate(delta_image_index=1)

    def _next_patch(self):
        self.navigate(delta_patch_index=1)

    def navigate(self, delta_image_index=0, delta_patch_index=0):
        if not len(self.acquisition_function.sampling_positions):
            return

        layer_names = list(self.acquisition_function.sampling_positions.keys())

        self._current_image += delta_image_index
        self._current_patch_position += delta_patch_index

        if delta_patch_index:
            if self._current_patch_position < 0:
                self._current_image -= 1

            elif (len(self.acquisition_function.sampling_positions[layer_names[self._current_image]])
                  <= self._current_patch_position):
                self._current_image += 1
                self._current_patch_position = 0

        if self._current_image < 0:
            self._current_image = len(layer_names) - 1

        elif self._current_image >= len(layer_names):
            self._current_image = 0

        if self._current_patch_position < 0:
            self._current_patch_position = len(
                self.acquisition_function.sampling_positions[layer_names[self._current_image]]
            ) - 1

        cur_position = (self.acquisition_function.sampling_positions
                        [layer_names[self._current_image]]
                        [self._current_patch_position])

        current_center = [(ax_roi.stop + ax_roi.start) / 2
                          for ax_roi in cur_position]

        self.viewer.dims.order = tuple(range(self.viewer.dims.ndim))
        self.viewer.camera.center = (
            *self.viewer.camera.center[:-len(current_center)],
            *current_center
        )
        self.viewer.camera.zoom = 2

    def fix_labels(self):
        layer_name = list(self.acquisition_function.sampling_positions.keys())[self._current_image]
        current_position = self.acquisition_function.sampling_positions[layer_name][self._current_patch_position]

        zarr_path = self.viewer.layers[layer_name]._source.path

        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': zarr_path,
            },
        }

        ts_array = ts.open(spec).result()

        self.txn = ts.Transaction()

        self.viewer.add_labels(ts_array.with_transaction(self.txn)[ts[current_position].translate_to[0]], name='paint', blending="opaque", opacity=0.7)
        self.viewer.layers['paint'].translate = [ax_roi.start for ax_roi in current_position]
        self.viewer.layers['paint'].bounding_box.visible = True


if __name__ == "__main__":
    viewer = napari.Viewer()

    mask_generator = MaskGenerator(viewer)
    viewer.window.add_dock_widget(mask_generator)

    acquisition_function = AcquisitionFunction(viewer)
    viewer.window.add_dock_widget(acquisition_function)

    metadata_manager = MetadataManager(viewer)
    viewer.window.add_dock_widget(metadata_manager)

    labels_manager_widget = LabelsManager(viewer, acquisition_function)
    viewer.window.add_dock_widget(labels_manager_widget, area='right')


    napari.run()
