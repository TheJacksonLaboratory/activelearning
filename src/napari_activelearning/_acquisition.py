from typing import Optional, Iterable, Tuple, Callable, Union
import random
from pathlib import Path
import numpy as np
import math

import zarrdataset as zds
import dask.array as da

try:
    import torch
    from torch.utils.data import DataLoader
    USING_PYTORCH = True
except ModuleNotFoundError:
    USING_PYTORCH = False

import napari
from napari.layers._multiscale_data import MultiScaleData

from ._layers import ImageGroupsManager, ImageGroup, LayersGroup
from ._labels import LabelsManager, LabelItem
from ._utils import get_dataloader, save_zarr, downsample_image


def compute_BALD(probs):
    if probs.ndim == 3:
        probs = np.stack((probs, 1 - probs), axis=1)

    T = probs.shape[0]

    probs_mean = probs.mean(axis=0)

    mutual_info = (-np.sum(probs_mean * np.log(probs_mean + 1e-12), axis=0)
                   + np.sum(probs * np.log(probs + 1e-12), axis=(0, 1)) / T)

    return mutual_info


def compute_acquisition_superpixel(probs, super_pixel_labels):
    mutual_info = compute_BALD(probs)

    super_pixel_indices = np.unique(super_pixel_labels)

    u_sp_lab = np.zeros_like(super_pixel_labels, dtype=np.float32)

    for sp_l in super_pixel_indices:
        mask = super_pixel_labels == sp_l
        u_val = np.sum(mutual_info[mask]) / np.sum(mask)
        u_sp_lab = np.where(mask, u_val, u_sp_lab)

    return u_sp_lab


def compute_acquisition_fun(tunable_segmentation_method, img, img_sp,
                            MC_repetitions):
    probs = []
    for _ in range(MC_repetitions):
        probs.append(
            tunable_segmentation_method.probs(img)
        )
    probs = np.stack(probs, axis=0)

    u_sp_lab = compute_acquisition_superpixel(probs, img_sp)

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
        contrast_limits: Optional[Iterable[float]] = None,
        colormap: Optional[str] = None,
        use_as_input_labels: bool = False,
        add_func: Optional[Callable] = napari.Viewer.add_image
):
    if output_filename:
        root = output_filename

    # Downsample the acquisition function
    output_fun_ms = downsample_image(
        root,
        source_axes=axes,
        data_group=data_group,
        scale=2,
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
        scale=list(scale.values()),
        translate=tuple(
            reference_scale.get(ax, 1) / 2.0
            if reference_scale.get(ax, 1) > 1 else 0
            for ax in reference_source_axes
            ),
        blending="translucent_no_depth",
    )

    if colormap is not None:
        func_args["colormap"] = colormap

    if contrast_limits is not None:
        func_args["contrast_limits"] = contrast_limits

    new_output_layer = add_func(**func_args)

    if isinstance(new_output_layer, list):
        new_output_layer = new_output_layer[0]

    output_layers_group = image_group.getLayersGroup(
        layers_group_name
    )

    if output_layers_group is None:
        output_layers_group = image_group.add_layers_group(
            layers_group_name,
            source_axes=axes,
            use_as_input_image=False,
            use_as_input_labels=use_as_input_labels,
            use_as_sampling_mask=False
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


class SegmentationMethod:
    def __init__(self):
        super().__init__()

    def _run_pred(self, img, *args, **kwargs):
        raise NotImplementedError("This method requies to be overriden by a "
                                  "derived class.")

    def _run_eval(self, img, *args, **kwargs):
        raise NotImplementedError("This method requies to be overriden by a "
                                  "derived class.")

    def probs(self, img, *args, **kwargs):
        probs = self._run_pred(img, *args, **kwargs)
        return probs

    def segment(self, img, *args, **kwargs):
        out = self._run_eval(img, *args, **kwargs)
        return out


class TunableMethod(SegmentationMethod):
    def __init__(self):
        self._num_workers = 0
        super().__init__()

    def _get_transform(self):
        raise NotImplementedError("This method requies to be overriden by a "
                                  "derived class.")

    def _fine_tune(self, train_data, train_labels, test_data, test_labels):
        raise NotImplementedError("This method requies to be overriden by a "
                                  "derived class.")

    def fine_tune(self, dataset_metadata_list: Iterable[dict],
                  train_data_proportion: float = 0.8,
                  patch_sizes: Union[dict, int] = 256,
                  model_axes="YXC"):
        train_data = []
        test_data = []
        train_labels = []
        test_labels = []

        transform = self._get_transform()

        for dataset_metadata in dataset_metadata_list:
            patch_sampler = zds.PatchSampler(
                patch_size=patch_sizes,
                spatial_axes=dataset_metadata["labels"]["axes"],
                min_area=0.01
            )

            dataset = zds.ZarrDataset(
                list(dataset_metadata.values()),
                return_positions=False,
                draw_same_chunk=False,
                patch_sampler=patch_sampler,
                shuffle=True,
            )

            dataset.add_transform("images", zds.ToDtype(np.float32))
            dataset.add_transform("labels", zds.ToDtype(np.int32))

            if USING_PYTORCH:
                dataloader = DataLoader(
                    dataset,
                    num_workers=self._num_workers,
                    worker_init_fn=zds.zarrdataset_worker_init_fn
                )
            else:
                dataloader = dataset

            drop_axis = tuple(
                ax_idx
                for ax_idx, ax in enumerate(
                    dataset_metadata["images"]["axes"])
                if ax != "C" and ax not in model_axes
            )

            drop_label_axis = tuple(
                ax_idx
                for ax_idx, ax in enumerate(
                    dataset_metadata["labels"]["axes"])
                if ax != "C" and ax not in model_axes
            )

            for img, lab in dataloader:
                if USING_PYTORCH:
                    img = img[0].numpy()
                    lab = lab[0].numpy()

                if len(drop_axis):
                    img = img.squeeze(drop_axis)

                if len(drop_label_axis):
                    lab = lab.squeeze(drop_label_axis)

                img = transform(img)

                if random.random() <= train_data_proportion:
                    train_data.append(img)
                    train_labels.append(lab)
                else:
                    test_data.append(img)
                    test_labels.append(lab)

        if not test_data:
            # Take at least one sample at random from the train dataset
            test_data_idx = random.randrange(0, len(train_data))
            test_data = [train_data.pop(test_data_idx)]
            test_labels = [train_labels.pop(test_data_idx)]

        self._fine_tune(train_data, train_labels, test_data, test_labels)

        return train_data, train_labels, test_data, test_labels


class AcquisitionFunction:
    def __init__(self, image_groups_manager: ImageGroupsManager,
                 labels_manager: LabelsManager,
                 tunable_segmentation_methods: dict):
        self._patch_sizes = {}
        self._max_samples = 1
        self._MC_repetitions = 3

        viewer = napari.current_viewer()
        self.input_axes = "".join(viewer.dims.axis_labels).upper()
        self.model_axes = "".join(viewer.dims.axis_labels).upper()

        self.image_groups_manager = image_groups_manager
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
                                    layers_group.shape)
            }

            dataset_metadata[layer_type] = layers_group.metadata
            dataset_metadata[layer_type]["roi"] = None

            if layer_type in ["images", "labels", "masks"]:
                dataset_metadata[layer_type]["roi"] = [tuple(
                    slice(0, math.ceil(ax_s / displayed_shape[ax]
                                       * (displayed_shape[ax]
                                          - displayed_shape[ax]
                                          % self._patch_sizes.get(ax, 1))))
                    if (ax in displayed_shape
                        and (ax in self.model_axes
                             or ax_s > self._patch_sizes.get(ax, 1)))
                    else slice(0, 1, None)
                    for ax, ax_s in layers_group_shape.items()
                )]

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
                    lambda ax: ax not in self.model_axes,
                    layers_group.source_axes
                ))

                model_spatial_axes += list(self.model_axes)
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
        else:
            self.tunable_segmentation_method = None

    def compute_acquisition(self, dataset_metadata, acquisition_fun,
                            segmentation_out,
                            sampled_mask=None,
                            segmentation_only=False):
        if self.tunable_segmentation_method is None:
            return

        model_spatial_axes = [
            ax
            for ax in self.model_axes
            if ax != "C"
        ]
        model_spatial_axes = "".join(model_spatial_axes)

        input_spatial_axes = [
            ax
            for ax in dataset_metadata["images"]["source_axes"]
            if ax in self.input_axes and ax != "C"
        ]
        input_spatial_axes = "".join(input_spatial_axes)

        output_axes = "TCZYX"

        dl = get_dataloader(dataset_metadata, patch_size=self._patch_sizes,
                            spatial_axes=input_spatial_axes,
                            model_input_axes=self.model_axes,
                            shuffle=True)
        segmentation_max = 0
        n_samples = 0
        img_sampling_positions = []

        pred_sel = tuple(
            slice(None) if ax in model_spatial_axes else None
            for ax in output_axes
        )

        drop_axis = tuple(
            ax_idx
            for ax_idx, ax in enumerate(
                dataset_metadata["images"]["axes"])
            if ax != "C" and ax not in model_spatial_axes
        )

        drop_axis_sp = list(drop_axis)
        if "C" in dataset_metadata["images"]["axes"]:
            drop_axis_sp.append(dataset_metadata["images"]["axes"].index("C"))
        drop_axis_sp = tuple(drop_axis_sp)

        self._reset_patch_progressbar()
        for pos, img, img_sp in dl:
            if USING_PYTORCH:
                pos = pos[0].numpy()
                img = img[0].numpy()
                img_sp = img_sp[0].numpy()

            img_shape = img.shape
            if len(drop_axis):
                img = img.squeeze(drop_axis)

            if len(drop_axis_sp):
                img_sp = img_sp.squeeze(drop_axis_sp)

            pos = {
                ax: slice(pos_ax[0],
                          pos_ax[1] if pos_ax[1] > 0 else ax_s)
                for ax, ax_s, pos_ax in zip(
                    dataset_metadata["images"]["axes"], img_shape, pos)
            }

            pos_u_lab = tuple(
                pos.get(ax, slice(0, 1))
                if ax != "C" else slice(0, 1)
                for ax in output_axes
            )

            if not segmentation_only:
                u_sp_lab = compute_acquisition_fun(
                    self.tunable_segmentation_method,
                    img,
                    img_sp,
                    self._MC_repetitions,
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

            if sampled_mask is not None:
                scaled_pos_u_lab = tuple(
                    slice(pos[ax].start
                          // self._patch_sizes.get(ax, 1),
                          pos[ax].stop
                          // self._patch_sizes.get(ax, 1))
                    if ax in pos and ax != "C" else slice(0, 1)
                    for ax in output_axes
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
        return img_sampling_positions

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
            sampling_mask_layers_group = None
            if image_group.sampling_mask_layers_group is not None:
                sampling_mask_layers_group = image_group.child(
                    image_group.sampling_mask_layers_group
                )

            displayed_source_axes = input_layers_group.source_axes
            displayed_shape = {
                ax: ax_s
                for ax, ax_s in zip(displayed_source_axes,
                                    input_layers_group.shape)
            }
            displayed_scale = {
                ax: ax_scl
                for ax, ax_scl in zip(displayed_source_axes,
                                      input_layers_group.scale)
            }

            output_axes = "TCZYX"
            output_shape = [
                displayed_shape.get(ax, 1)
                if ax != "C" else 1
                for ax in output_axes
            ]

            if not segmentation_only:
                acquisition_root = save_zarr(
                    output_filename,
                    data=None,
                    shape=output_shape,
                    chunk_size=True,
                    name="acquisition_fun",
                    dtype=np.float32,
                    is_label=False,
                    is_multiscale=True
                )

                acquisition_fun_grp = acquisition_root[""
                                                       "acquisition_fun/0"]
            else:
                acquisition_fun_grp = None

            segmentation_root = save_zarr(
                output_filename,
                data=None,
                shape=output_shape,
                chunk_size=True,
                name=segmentation_group_name,
                dtype=np.int32,
                is_label=True,
                is_multiscale=True
            )

            segmentation_grp = segmentation_root[
                f"{segmentation_group_name}/0"
            ]

            dataset_metadata = self._prepare_datasets_metadata(
                 displayed_shape,
                 [(input_layers_group, "images"),
                  (sampling_mask_layers_group, "masks")]
                )

            if "sampled_positions" not in segmentation_root.keys():
                sampling_output_shape = [
                    math.ceil(displayed_shape.get(ax, 1)
                              // self._patch_sizes.get(ax, 1))
                    if ax != "C" else 1
                    for ax in output_axes
                ]
                sampling_output_scale = {
                    ax: (displayed_scale.get(ax, 1)
                         * self._patch_sizes.get(ax, 1))
                    for ax in output_axes
                }

                sampled_root = save_zarr(
                    output_filename,
                    data=None,
                    shape=sampling_output_shape,
                    chunk_size=True,
                    name="sampled_positions",
                    dtype=np.uint8,
                    is_label=True,
                    is_multiscale=True
                )

                sampled_grp = sampled_root["sampled_positions/0"]
            else:
                sampled_grp = None

            # Compute acquisition function of the current image
            img_sampling_positions = self.compute_acquisition(
                dataset_metadata,
                acquisition_fun=acquisition_fun_grp,
                segmentation_out=segmentation_grp,
                sampled_mask=sampled_grp,
                segmentation_only=segmentation_only
            )

            self._update_image_progressbar(n + 1)

            if not img_sampling_positions:
                continue

            if not segmentation_only:
                add_multiscale_output_layer(
                    acquisition_root,
                    axes=output_axes,
                    scale=displayed_scale,
                    data_group="acquisition_fun/0",
                    group_name=group_name + " acquisition function",
                    layers_group_name="acquisition",
                    image_group=image_group,
                    reference_source_axes=displayed_source_axes,
                    reference_scale=displayed_scale,
                    output_filename=output_filename,
                    contrast_limits=(
                        0, max(img_sampling_positions).acquisition_val
                    ),
                    colormap="magma",
                    add_func=viewer.add_image
                )

            if sampled_grp is not None:
                add_multiscale_output_layer(
                    sampled_root,
                    axes=output_axes,
                    scale=sampling_output_scale,
                    data_group="sampled_positions/0",
                    group_name=group_name + " sampled positions",
                    layers_group_name="sampled positions",
                    image_group=image_group,
                    reference_source_axes=output_axes,
                    reference_scale=sampling_output_scale,
                    output_filename=output_filename,
                    add_func=viewer.add_labels
                )

            segmentation_channel = add_multiscale_output_layer(
                segmentation_root,
                axes=output_axes,
                scale=displayed_scale,
                data_group=f"{segmentation_group_name}/0",
                group_name=group_name + f" {segmentation_group_name}",
                layers_group_name=segmentation_group_name,
                image_group=image_group,
                reference_source_axes=displayed_source_axes,
                reference_scale=displayed_scale,
                output_filename=output_filename,
                use_as_input_labels=True,
                add_func=viewer.add_labels
            )

            if (not segmentation_only
               and image_group is not None
               and image_group.labels_group is None):
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

            if (sampling_mask_layers_group is not None
               and image_group.labels_group is None):
                layer_types.append((sampling_mask_layers_group, "masks"))

            displayed_source_axes = input_layers_group.source_axes
            displayed_shape = {
                ax: ax_s
                for ax, ax_s in zip(displayed_source_axes,
                                    input_layers_group.shape)
            }

            output_axes = displayed_source_axes
            if "C" in output_axes:
                output_axes = list(output_axes)
                output_axes.remove("C")
                output_axes = "".join(output_axes)

            dataset_metadata = self._prepare_datasets_metadata(
                 displayed_shape,
                 layer_types,
                )

            dataset_metadata_list.append(dataset_metadata)

        self.tunable_segmentation_method.fine_tune(
            dataset_metadata_list,
            patch_sizes=self._patch_sizes,
            model_axes=self.model_axes
        )

        self.compute_acquisition_layers(
            run_all=True,
            segmentation_group_name="fine_tunned_segmentation",
            segmentation_only=True
        )

        return True
