from typing import Optional, Iterable, Tuple, Callable

try:
    import torch
    USING_PYTORCH = True
except ModuleNotFoundError:
    USING_PYTORCH = False

import os
from pathlib import Path
import numpy as np
import math
import dask.array as da
import zarr
import napari
from napari.layers._multiscale_data import MultiScaleData

from ._layers import ImageGroupsManager, ImageGroup, LayersGroup
from ._labels import LabelsManager, LabelItem
from ._utils import get_dataloader, save_zarr, downsample_image, update_labels


def compute_BALD(probs):
    if probs.ndim == 3:
        probs = np.stack((probs, 1 - probs), axis=1)

    T = probs.shape[0]

    probs_mean = probs.mean(axis=0)

    mutual_info = (-np.sum(probs_mean * np.log(probs_mean + 1e-12), axis=0)
                   + np.sum(probs * np.log(probs + 1e-12), axis=(0, 1)) / T)

    return mutual_info


def compute_acquisition_superpixel(mutual_info, super_pixel_labels):
    super_pixel_indices = np.unique(super_pixel_labels)

    u_sp_lab = np.zeros_like(super_pixel_labels, dtype=np.float32)

    for sp_l in super_pixel_indices:
        mask = super_pixel_labels == sp_l
        u_val = np.sum(mutual_info[mask]) / np.sum(mask)
        if np.isnan(u_val):
            u_val = 0.0

        u_sp_lab = np.where(mask, u_val, u_sp_lab)

    return u_sp_lab


def compute_acquisition_fun(tunable_segmentation_method, img, MC_repetitions,
                            img_superpixel=None):
    probs = []
    for _ in range(MC_repetitions):
        probs.append(
            tunable_segmentation_method.probs(img)
        )
    probs = np.stack(probs, axis=0)

    mutual_info = compute_BALD(probs)
    if img_superpixel is not None:
        u_sp_lab = compute_acquisition_superpixel(mutual_info, img_superpixel)
    else:
        u_sp_lab = mutual_info

    return u_sp_lab


def compute_segmentation(tunable_segmentation_method, img, labels_offset=0):
    seg_out = tunable_segmentation_method.segment(img)
    seg_out = np.where(seg_out, seg_out + labels_offset, 0)
    return seg_out


def add_multiscale_output_layer(
        root,
        axes: str,
        scale: dict,
        data_group: str,
        group_name: str,
        layers_group_name: str,
        image_group: ImageGroup,
        reference_source_axes: str,
        reference_scale: dict,
        output_filename: Optional[Path] = None,
        colormap: Optional[str] = None,
        use_as_input_labels: bool = False,
        use_as_sampling_mask: bool = False,
        add_func: Optional[Callable] = napari.Viewer.add_image
):
    if output_filename:
        root = output_filename

    # Downsample the acquisition function
    output_fun_ms = downsample_image(
        root,
        axes=axes,
        scale=scale,
        data_group=data_group,
        downsample_scale=2,
        num_scales=5,
        reference_source_axes=reference_source_axes,
        reference_scale=reference_scale
    )

    is_multiscale = False
    if len(output_fun_ms) > 1:
        is_multiscale = True
    else:
        output_fun_ms = output_fun_ms[0]

    func_args = dict(
        data=output_fun_ms,
        name=group_name,
        multiscale=is_multiscale,
        opacity=0.8,
        scale=list(
            reference_scale.get(ax, 1) * scale.get(ax, 1)
            for ax in axes
        ),
        translate=tuple(
            (reference_scale.get(ax, 1) * scale.get(ax, 1) - 1) / 2.0
            if (reference_scale.get(ax, 1) * scale.get(ax, 1)) > 1 else 0
            for ax in axes
            ),
        blending="translucent_no_depth",
    )

    if colormap is not None:
        func_args["colormap"] = colormap

    new_output_layer = add_func(**func_args)

    if isinstance(new_output_layer, list):
        new_output_layer = new_output_layer[0]

    output_layers_group = image_group.add_layers_group(
        layers_group_name,
        source_axes=axes,
        use_as_input_image=False,
        use_as_input_labels=use_as_input_labels,
        use_as_sampling_mask=use_as_sampling_mask
    )

    output_channel = output_layers_group.add_layer(
        new_output_layer
    )

    if output_filename:
        output_channel.source_data = str(output_filename)
        output_channel.data_group = data_group

    return output_channel


if USING_PYTORCH:
    class DropoutEvalOverrider(torch.nn.Module):
        def __init__(self, dropout_module):
            super(DropoutEvalOverrider, self).__init__()

            self._dropout = type(dropout_module)(
                dropout_module.p, inplace=dropout_module.inplace)

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
                    if isinstance(layer, (torch.nn.Threshold,
                                          torch.nn.ReLU,
                                          torch.nn.RReLU,
                                          torch.nn.Hardtanh,
                                          torch.nn.ReLU6,
                                          torch.nn.Sigmoid,
                                          torch.nn.Hardsigmoid,
                                          torch.nn.Tanh,
                                          torch.nn.SiLU,
                                          torch.nn.Mish,
                                          torch.nn.Hardswish,
                                          torch.nn.ELU,
                                          torch.nn.CELU,
                                          torch.nn.SELU,
                                          torch.nn.GLU,
                                          torch.nn.GELU,
                                          torch.nn.Hardshrink,
                                          torch.nn.LeakyReLU,
                                          torch.nn.LogSigmoid,
                                          torch.nn.Softplus,
                                          torch.nn.Softshrink,
                                          torch.nn.MultiheadAttention,
                                          torch.nn.PReLU,
                                          torch.nn.Softsign,
                                          torch.nn.Tanhshrink,
                                          torch.nn.Softmin,
                                          torch.nn.Softmax,
                                          torch.nn.Softmax2d,
                                          torch.nn.LogSoftmax)):
                        break
                else:
                    continue

                dropout_layer = torch.nn.Dropout(p=p, inplace=True)
                module.insert(l_idx + 1, DropoutEvalOverrider(dropout_layer))

else:
    def add_dropout(net, p=0.05):
        pass


class AcquisitionFunction:
    def __init__(self, image_groups_manager: ImageGroupsManager,
                 labels_manager: LabelsManager,
                 tunable_segmentation_methods: dict):
        self._patch_sizes = {}
        self._previous_patch_sizes = None
        self.input_axes = ""

        self._max_samples = 1
        self._prev_max_samples = 0
        self._MC_repetitions = 3
        self._add_padding = False
        self._padding_factor = 4
        self._num_workers = 0

        self.image_groups_manager = image_groups_manager
        self.image_groups_manager.register_listener(self)

        self.labels_manager = labels_manager
        self.tunable_segmentation_method = None
        self._tunable_segmentation_methods = tunable_segmentation_methods

        super().__init__()

    def _reset_image_progressbar(self, num_images: int):
        pass

    def _update_image_progressbar(self, curr_image_index: int):
        pass

    def _reset_patch_progressbar(self):
        pass

    def _update_patch_progressbar(self, curr_patch_index: int):
        pass

    def _prepare_datasets_metadata(
            self,
            displayed_shape: dict,
            displayed_scale: dict,
            layer_types: Iterable[Tuple[LayersGroup, str]]):
        dataset_metadata = {}

        reference_axes = None

        # Make sure that the "images" layers are prepared first, so the
        # reference axes are computed from it.
        layers_groups, layers_types = list(zip(*layer_types))
        layers_types = list(layers_types)
        layers_groups = list(layers_groups)
        images_idx = layers_types.index("images")

        layers_types = [layers_types.pop(images_idx)] + layers_types
        layers_groups = [layers_groups.pop(images_idx)] + layers_groups

        for layers_group, layer_type in zip(layers_groups, layers_types):
            if layers_group is None:
                continue

            layers_group_shape = {
                ax: ax_s
                for ax, ax_s in zip(layers_group.source_axes,
                                    layers_group.selected_level_shape)
            }

            dataset_metadata[layer_type] = layers_group.metadata
            dataset_metadata[layer_type]["roi"] = None

            scaled_patch_sizes = {
                ax: ax_ps // displayed_scale.get(ax, 1)
                for ax, ax_ps in self._patch_sizes.items()
            }

            if layer_type in ["images", "labels", "masks"]:
                try:
                    dataset_metadata[layer_type]["roi"] = [tuple(
                        slice(0,
                              math.ceil(ax_s / displayed_shape[ax]
                                        * (displayed_shape[ax]
                                           - displayed_shape[ax]
                                           % scaled_patch_sizes.get(ax, 1))))
                        if (ax in displayed_shape
                            and (ax in self.tunable_segmentation_method
                                           .model_axes
                                 or ax_s > scaled_patch_sizes.get(ax, 1)))
                        else slice(None)
                        for ax, ax_s in layers_group_shape.items()
                    )]
                except TypeError as err:
                    print(f"Error: {err}")

            if isinstance(dataset_metadata[layer_type]["filenames"],
                          MultiScaleData):
                dataset_metadata[layer_type]["filenames"] =\
                    dataset_metadata[layer_type]["filenames"][0]

            if isinstance(dataset_metadata[layer_type]["filenames"],
                          da.core.Array):
                dataset_metadata[layer_type]["filenames"] =\
                    dataset_metadata[layer_type]["filenames"].compute()

            dataset_metadata[layer_type]["modality"] = layer_type

            if reference_axes is None:
                # Add axes that are not used by the model.
                # These are removed later in the acquisition function.
                model_spatial_axes = list(filter(
                    lambda ax:
                    ax not in self.tunable_segmentation_method.model_axes,
                    layers_group.source_axes
                ))

                model_spatial_axes += list(
                    self.tunable_segmentation_method.model_axes
                )
                model_spatial_axes = "".join(model_spatial_axes)
                reference_axes = model_spatial_axes

            else:
                model_spatial_axes = list(reference_axes)
                if "C" in reference_axes:
                    model_spatial_axes.remove("C")
                model_spatial_axes = "".join(model_spatial_axes)

            dataset_metadata[layer_type]["axes"] = model_spatial_axes

        return dataset_metadata

    def set_model(self, selected_model):
        tunable_segmentation_method_cls =\
            self._tunable_segmentation_methods.get(selected_model, None)

        if tunable_segmentation_method_cls is not None:
            self.tunable_segmentation_method =\
                tunable_segmentation_method_cls()
            self.tunable_segmentation_method.max_samples_per_image =\
                self._max_samples
        else:
            self.tunable_segmentation_method = None

    def update_reference_info(self):
        self.input_axes = []
        patch_sizes = {}

        for idx in range(self.image_groups_manager.groups_root.childCount()):
            child = self.image_groups_manager.groups_root.child(idx)
            if isinstance(child, ImageGroup):
                image_group = child
            else:
                continue

            input_layers_group_idx = image_group.input_layers_group
            if input_layers_group_idx is None:
                continue

            input_layers_group = image_group.child(input_layers_group_idx)
            input_layers_source_axes = input_layers_group.source_axes
            input_layers_shapes = input_layers_group.selected_level_shape

            self.input_axes += [
                ax
                for ax in input_layers_source_axes
                if ax not in self.input_axes
            ]

            patch_sizes.update({
                ax: min(128, ax_ps) if ax_ps else 128
                for ax, ax_ps in zip(input_layers_source_axes,
                                     input_layers_shapes)
            })

        if "C" in patch_sizes:
            del patch_sizes["C"]

        if self._previous_patch_sizes is not None:
            patch_sizes = {
                ax: self._previous_patch_sizes.get(ax, ps)
                for ax, ps in patch_sizes.items()
            }

        self._patch_sizes = patch_sizes
        self.input_axes = "".join([
            ax
            for ax in self.input_axes
            if ax != "C"
        ])

    def compute_acquisition(self, dataset_metadata, output_axes, mask_axes,
                            reference_scale,
                            acquisition_fun,
                            segmentation_out,
                            sampled_mask=None,
                            segmentation_only=False):
        if self.tunable_segmentation_method is None:
            return

        model_spatial_axes = "".join([
            ax
            for ax in self.tunable_segmentation_method.model_axes
            if ax != "C"
        ])

        batch_axes = dataset_metadata["images"]["axes"]

        input_spatial_axes = "".join([
            ax
            for ax in dataset_metadata["images"]["source_axes"]
            if ax in self.input_axes and ax != "C"
        ])

        scaled_patch_sizes = {
            ax: ax_ps // reference_scale.get(ax, 1)
            for ax, ax_ps in self._patch_sizes.items()
        }

        padding = {}
        if self._add_padding:
            padding = {
                ax: ax_ps // self._padding_factor
                for ax, ax_ps in scaled_patch_sizes.items()
            }

        dl = get_dataloader(
            dataset_metadata,
            patch_size=scaled_patch_sizes,
            spatial_axes=input_spatial_axes,
            padding=padding,
            model_input_axes=self.tunable_segmentation_method.model_axes,
            shuffle=True,
            num_workers=self._num_workers,
            tunable_segmentation_method=self.tunable_segmentation_method
        )

        segmentation_max = 0
        unique_labels = set()
        n_samples = 0
        img_sampling_positions = []

        pred_sel = tuple(
            slice(padding.get(ax, 0)
                  if padding.get(ax, 0) > 0 else None,
                  scaled_patch_sizes.get(ax, 0)
                  + padding[ax] if padding.get(ax, 0) > 0 else None)
            if ax in model_spatial_axes else None
            for ax in output_axes
        )

        drop_axis_sp = []
        if "C" in dataset_metadata["images"]["axes"]:
            drop_axis_sp.append(
                self.tunable_segmentation_method.model_axes.index("C")
            )
        drop_axis_sp = tuple(drop_axis_sp)

        self._reset_patch_progressbar()
        for pos, img, img_sp in dl:
            if USING_PYTORCH:
                pos = pos[0].numpy()
                img = img[0].numpy()
                img_sp = img_sp[0].numpy()

            if len(drop_axis_sp):
                img_sp = img_sp.squeeze(drop_axis_sp)

            pos_axes = {
                ax: pos_ax
                for ax, pos_ax in zip(batch_axes, pos)
            }

            img_shape = {
                ax: ax_s
                for ax, ax_s in zip(
                    self.tunable_segmentation_method.model_axes,
                    img.shape)
            }

            pos_padded = {
                ax: slice(pos_axes.get(ax, (0, 0))[0] + padding.get(ax, 0),
                          pos_axes.get(ax, (0, 0))[1] - padding.get(ax, 0)
                          if pos_axes.get(ax, (0, 0))[1] > 0 else
                          img_shape.get(ax, 1))
                if (ax != "C"
                    or ax in self.tunable_segmentation_method.model_axes)
                else slice(0, 1)
                for ax in output_axes
            }

            pos_u_lab = tuple(
                pos_padded.get(ax, slice(0, 1))
                if ax != "C" else slice(0, 1)
                for ax in output_axes
            )

            if not segmentation_only:
                u_sp_lab = compute_acquisition_fun(
                    self.tunable_segmentation_method,
                    img,
                    self._MC_repetitions,
                    # img_superpixel=img_sp,
                )

                acquisition_fun[pos_u_lab] = u_sp_lab[pred_sel]
                acquisition_val = u_sp_lab.max()
            else:
                acquisition_val = 0

            seg_out = compute_segmentation(
                self.tunable_segmentation_method,
                img,
                segmentation_max
            )
            segmentation_out[pos_u_lab] = seg_out[pred_sel]
            segmentation_max = max(segmentation_max, seg_out.max())
            unique_labels = unique_labels.union(
                set(np.unique(seg_out[pred_sel]))
            )

            if sampled_mask is not None:
                scaled_pos_u_lab = tuple(
                    slice(pos_padded[ax].start
                          // scaled_patch_sizes.get(ax, 1),
                          pos_padded[ax].stop
                          // scaled_patch_sizes.get(ax, 1))
                    for ax in mask_axes
                )
                sampled_mask[scaled_pos_u_lab] = True

            img_sampling_positions.append(
                LabelItem(acquisition_val, position=pos_u_lab)
            )

            n_samples += 1
            if n_samples >= self._max_samples:
                break

            self._update_patch_progressbar(n_samples)

        self._update_patch_progressbar(self._max_samples)
        return img_sampling_positions, unique_labels

    def compute_acquisition_layers(
            self,
            run_all: bool = False,
            segmentation_group_name: Optional[str] = "segmentation",
            segmentation_only: bool = False,
            ):
        if self.tunable_segmentation_method is None:
            return

        if run_all:
            for idx in range(self.image_groups_manager.groups_root.childCount()
                             ):
                child = self.image_groups_manager.groups_root.child(idx)
                child.setSelected(isinstance(child, ImageGroup))

        image_groups = list(filter(
            lambda item:
            isinstance(item, ImageGroup),
            self.image_groups_manager.get_active_item()
        ))

        if not image_groups:
            return False

        self._reset_image_progressbar(len(image_groups))

        viewer = napari.current_viewer()
        for n, image_group in enumerate(image_groups):
            self.image_groups_manager.set_active_item(image_group)
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
            sampling_mask_layers_group = None
            mask_axes = None
            if image_group.sampling_mask_layers_group is not None:
                sampling_mask_layers_group = image_group.child(
                    image_group.sampling_mask_layers_group
                )
                mask_axes = sampling_mask_layers_group.source_axes

            displayed_source_axes = input_layers_group.source_axes
            displayed_shape = {
                ax: ax_s
                for ax, ax_s in zip(displayed_source_axes,
                                    input_layers_group.selected_level_shape)
            }
            displayed_reference_scale = {
                ax: ax_scl
                for ax, ax_scl in zip(displayed_source_axes,
                                      input_layers_group.scale)
            }
            displayed_scale = {
                ax: ax_scl
                for ax, ax_scl in zip(displayed_source_axes,
                                      input_layers_group.selected_level_scale)
            }

            dataset_metadata = self._prepare_datasets_metadata(
                 displayed_shape,
                 displayed_scale,
                 [(input_layers_group, "images"),
                  (sampling_mask_layers_group, "masks")]
            )

            output_scale = dict(displayed_scale)

            output_axes = displayed_source_axes
            if "C" in output_axes:
                output_axes = list(output_axes)
                output_axes.remove("C")
                output_axes = "".join(output_axes)

            if "C" in output_scale:
                output_scale.pop("C")

            output_shape = [
                displayed_shape.get(ax, 1)
                for ax in output_axes
                if ax != "C"
            ]

            if not segmentation_only:
                acquisition_root, acquisition_fun_grp_name = save_zarr(
                    output_filename,
                    data=None,
                    shape=output_shape,
                    chunk_size=True,
                    name="acquisition_fun",
                    dtype=np.float32,
                    is_label=False,
                    is_multiscale=True,
                    overwrite=False
                )
                acquisition_fun_grp =\
                    acquisition_root[f"{acquisition_fun_grp_name}/0"]

            else:
                acquisition_fun_grp = None

            segmentation_root, segmentation_group_name = save_zarr(
                output_filename,
                data=None,
                shape=output_shape,
                chunk_size=True,
                name=segmentation_group_name,
                dtype=np.int32,
                is_label=True,
                is_multiscale=True,
                overwrite=False
            )

            segmentation_grp =\
                segmentation_root[f"{segmentation_group_name}/0"]

            if sampling_mask_layers_group is None:
                sampling_output_scale = {
                    ax: int(displayed_scale.get(ax, 1)
                            * self._patch_sizes.get(ax, 1))
                    for ax in output_axes
                }
                self.image_groups_manager.mask_generator\
                                         .update_reference_info()
                self.image_groups_manager.mask_generator.set_patch_size(
                    [
                        sampling_output_scale.get(ax, 1)
                        for ax in self.image_groups_manager.mask_generator
                                                           ._mask_axes
                    ]
                )

                self.image_groups_manager.mask_generator.generate_mask_layer()

                sampling_mask_layers_group = image_group.child(
                    image_group.sampling_mask_layers_group
                )

                if isinstance(sampling_mask_layers_group.source_data,
                              (str, Path)):
                    sampled_grp = zarr.open(
                        os.path.join(
                            sampling_mask_layers_group.source_data,
                            sampling_mask_layers_group.data_group
                        ),
                        mode="r+"
                    )

                else:
                    sampled_grp = sampling_mask_layers_group.source_data

                mask_axes = sampling_mask_layers_group.source_axes
            else:
                sampled_grp = None

            # Compute acquisition function of the current image
            img_sampling_positions, unique_labels = self.compute_acquisition(
                dataset_metadata,
                output_axes=output_axes,
                mask_axes=mask_axes,
                reference_scale=displayed_scale,
                acquisition_fun=acquisition_fun_grp,
                segmentation_out=segmentation_grp,
                sampled_mask=sampled_grp,
                segmentation_only=segmentation_only,
            )

            if (sampled_grp is not None
               and sampling_mask_layers_group is not None):
                sampling_mask_layers_group.child(0).layer.refresh()

            self._update_image_progressbar(n + 1)

            if not img_sampling_positions:
                continue

            if not segmentation_only:
                add_multiscale_output_layer(
                    acquisition_root,
                    axes=output_axes,
                    scale=output_scale,
                    data_group=str(Path(acquisition_fun_grp_name) / "0"),
                    group_name=group_name + " acquisition function",
                    layers_group_name="acquisition",
                    image_group=image_group,
                    reference_source_axes=displayed_source_axes,
                    reference_scale=displayed_reference_scale,
                    output_filename=output_filename,
                    colormap="magma",
                    add_func=viewer.add_image
                )

            update_labels(
                segmentation_root[f"{segmentation_group_name}"],
                unique_labels
            )

            segmentation_channel = add_multiscale_output_layer(
                segmentation_root,
                axes=output_axes,
                scale=output_scale,
                data_group=str(Path(segmentation_group_name) / "0"),
                group_name=group_name + f" {segmentation_group_name}",
                layers_group_name=segmentation_group_name,
                image_group=image_group,
                reference_source_axes=displayed_source_axes,
                reference_scale=displayed_reference_scale,
                output_filename=output_filename,
                use_as_input_labels=False,
                add_func=viewer.add_labels
            )

            if (not segmentation_only
               and image_group is not None):
                new_label_group = self.labels_manager.add_labels(
                    segmentation_channel,
                    img_sampling_positions
                )

                image_group.labels_group = new_label_group

        return True

    def fine_tune(self):
        image_groups = list(filter(
            lambda item:
            isinstance(item, ImageGroup),
            map(lambda idx:
                self.image_groups_manager.groups_root.child(idx),
                range(self.image_groups_manager.groups_root.childCount()))
        ))
        if self.tunable_segmentation_method is None:
            return

        if not image_groups:
            return False

        dataset_metadata_list = []
        scaled_patch_sizes = dict(self._patch_sizes)

        for image_group in image_groups:
            image_group.setSelected(True)

            input_layers_group_idx = image_group.input_layers_group
            label_layers_group_idx = image_group.labels_layers_group

            if (input_layers_group_idx is None
               or label_layers_group_idx is None):
                continue

            sampling_mask_layers_group = None
            if image_group.sampling_mask_layers_group is not None:
                sampling_mask_layers_group = image_group.child(
                    image_group.sampling_mask_layers_group
                )

            input_layers_group = image_group.child(input_layers_group_idx)
            label_layers_group = image_group.child(label_layers_group_idx)

            layer_types = [
                (input_layers_group, "images"),
                (label_layers_group, "labels")
            ]

            displayed_source_axes = input_layers_group.source_axes
            displayed_shape = {
                ax: ax_s
                for ax, ax_s in zip(displayed_source_axes,
                                    input_layers_group.selected_level_shape)
            }
            displayed_scale = {
                ax: ax_scl
                for ax, ax_scl in zip(displayed_source_axes,
                                      input_layers_group.selected_level_scale)
            }

            output_axes = displayed_source_axes
            if "C" in output_axes:
                output_axes = list(output_axes)
                output_axes.remove("C")
                output_axes = "".join(output_axes)

            scaled_patch_sizes = {
                ax: ax_ps // displayed_scale.get(ax, 1)
                for ax, ax_ps in self._patch_sizes.items()
            }

            if sampling_mask_layers_group is not None:
                layer_types.append((sampling_mask_layers_group, "masks"))
            else:
                self.image_groups_manager.mask_generator.active_image_group =\
                    image_group
                self.image_groups_manager.mask_generator.set_patch_size(
                    [scaled_patch_sizes[ax] for ax in output_axes]
                )

                self.image_groups_manager.mask_generator.generate_mask_layer()

                sampling_mask_layers_group = image_group.child(
                    image_group.sampling_mask_layers_group
                )
                sampling_mask_layers_group.child(0).layer.data[:] = 1

                layer_types.append((sampling_mask_layers_group, "masks"))

            dataset_metadata = self._prepare_datasets_metadata(
                 displayed_shape,
                 displayed_scale,
                 layer_types,
                )

            dataset_metadata_list.append(dataset_metadata)

        success = self.tunable_segmentation_method.fine_tune(
            dataset_metadata_list,
            model_axes=self.tunable_segmentation_method.model_axes,
            patch_sizes=scaled_patch_sizes,
            num_workers=self._num_workers
        )

        self.compute_acquisition_layers(
            run_all=True,
            segmentation_group_name="fine_tunned_segmentation",
            segmentation_only=True
        )

        return success
