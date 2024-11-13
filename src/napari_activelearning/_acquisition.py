from typing import Optional, Iterable, Tuple, Callable, Union
from functools import partial
import random
from pathlib import Path
import numpy as np

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
from ._utils import (get_dataloader, save_zarr, downsample_image,
                     StaticPatchSampler)


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
        scale: Iterable[float],
        data_group: str,
        group_name: str,
        layers_group_name: str,
        image_group: ImageGroup,
        reference_source_axes: str,
        reference_scale: Iterable[float],
        output_filename: Optional[Path] = None,
        contrast_limits: Optional[Iterable[float]] = None,
        colormap: Optional[str] = None,
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

    func_args = dict(
        data=output_fun_ms,
        name=group_name,
        multiscale=True,
        opacity=0.8,
        scale=scale,
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
                    if isinstance(layer, torch.nn.ReLU):
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


class FineTuningMethod:
    def __init__(self):
        self._num_workers = 0
        super().__init__()

    def _get_transform(self):
        raise NotImplementedError("This method requies to be overriden by a "
                                  "derived class.")

    def _fine_tune(self, train_data, train_labels, test_data, test_labels):
        raise NotImplementedError("This method requies to be overriden by a "
                                  "derived class.")

    def fine_tune(self, dataset_metadata_list: Iterable[
        Tuple[dict, Iterable[Iterable[int]]]],
                  train_data_proportion: float = 0.8,
                  patch_sizes: Union[dict, int] = 256,
                  model_axes="YXC"):
        train_data = []
        test_data = []
        train_labels = []
        test_labels = []

        transform = self._get_transform()

        for dataset_metadata, top_lefts in dataset_metadata_list:
            dataset = zds.ZarrDataset(
                list(dataset_metadata.values()),
                return_positions=False,
                draw_same_chunk=False,
                patch_sampler=StaticPatchSampler(
                    patch_size=patch_sizes,
                    top_lefts=top_lefts,
                    spatial_axes=dataset_metadata["labels"]["axes"]
                ),
                shuffle=True,
            )

            dataset.add_transform("images", zds.ToDtype(np.float32))

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

            for img, lab in dataloader:
                if USING_PYTORCH:
                    img = img[0].numpy()
                    lab = lab[0].numpy()

                if len(drop_axis):
                    img = img.squeeze(drop_axis)
                    lab = lab.squeeze(drop_axis)

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


class TunableMethod(SegmentationMethod, FineTuningMethod):
    def __init__(self):
        super().__init__()


class AcquisitionFunction:
    def __init__(self, image_groups_manager: ImageGroupsManager,
                 labels_manager: LabelsManager,
                 tunable_segmentation_method: TunableMethod):
        self._patch_sizes = {}
        self._max_samples = 1
        self._MC_repetitions = 3

        viewer = napari.current_viewer()
        self.input_axes = "".join(viewer.dims.axis_labels).upper()
        self.model_axes = "".join(viewer.dims.axis_labels).upper()

        self.image_groups_manager = image_groups_manager
        self.labels_manager = labels_manager
        self.tunable_segmentation_method = tunable_segmentation_method

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
            image_group: ImageGroup,
            output_axes: str,
            displayed_source_axes: str,
            displayed_shape: Iterable[int],
            layer_types: Iterable[Tuple[LayersGroup, str]]):
        dataset_metadata = {}
        sampling_positions = None

        for layers_group, layer_type in layer_types:
            if layers_group is None:
                continue

            dataset_metadata[layer_type] = layers_group.metadata
            dataset_metadata[layer_type]["roi"] = None

            if layer_type in ["images", "labels"]:
                dataset_metadata[layer_type]["roi"] = [tuple(
                    slice(0, ax_s - ax_s % self._patch_sizes.get(ax, 1))
                    if (ax != "C"
                        and (ax in self.model_axes
                             or ax_s > self._patch_sizes.get(ax, 1)))
                    else slice(None)
                    for ax, ax_s in zip(displayed_source_axes,
                                        displayed_shape)
                    if (layer_type == "images"
                        or (layer_type == "labels" and ax != "C"))
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

            model_spatial_axes = list(filter(
                lambda ax: ax not in self.model_axes,
                dataset_metadata[layer_type]["source_axes"]
            ))

            model_spatial_axes += list(self.model_axes)
            if "images" not in layer_type and "C" in model_spatial_axes:
                model_spatial_axes.remove("C")

            model_spatial_axes = "".join(model_spatial_axes)

            dataset_metadata[layer_type]["axes"] = model_spatial_axes

            if "images" in layer_type and image_group.labels_group:
                # Remove non-input axes from sampled positions
                labels = map(
                    lambda idx: image_group.labels_group.child(idx),
                    range(image_group.labels_group.childCount())
                )

                spatial_pos = map(
                    lambda child: [
                        ax_pos.start
                        for ax, ax_pos in zip(output_axes, child.position)
                        if ax in model_spatial_axes
                    ],
                    labels
                )

                sampling_positions = list(spatial_pos)

        return dataset_metadata, sampling_positions

    def compute_acquisition(self, dataset_metadata, acquisition_fun,
                            segmentation_out,
                            sampling_positions=None,
                            segmentation_only=False):
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

        dl = get_dataloader(dataset_metadata, patch_size=self._patch_sizes,
                            sampling_positions=sampling_positions,
                            spatial_axes=input_spatial_axes,
                            model_input_axes=self.model_axes,
                            shuffle=True)
        segmentation_max = 0
        n_samples = 0
        img_sampling_positions = []

        pred_sel = tuple(
            slice(None) if ax in model_spatial_axes else None
            for ax in input_spatial_axes
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

            if len(drop_axis):
                img = img.squeeze(drop_axis)

            if len(drop_axis_sp):
                img_sp = img_sp.squeeze(drop_axis_sp)

            pos = {
                ax: slice(pos_ax[0], pos_ax[1])
                for ax, pos_ax in zip(
                    dataset_metadata["images"]["axes"], pos)
            }

            pos_u_lab = tuple(pos[ax] for ax in input_spatial_axes)

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
            displayed_shape = input_layers_group.shape
            displayed_scale = input_layers_group.scale

            (output_axes,
             output_shape,
             output_scale) = list(zip(*[
                 (ax, ax_s, ax_scl)
                 for ax, ax_s, ax_scl in zip(displayed_source_axes,
                                             displayed_shape,
                                             displayed_scale)
                 if ax != "C"
                 ]))

            output_axes = "".join(output_axes)

            if not segmentation_only:
                acquisition_root = save_zarr(
                    output_filename,
                    data=None,
                    shape=output_shape,
                    chunk_size=True,
                    name="acquisition_fun",
                    dtype=np.float32,
                    is_label=True,
                    is_multiscale=True
                )

                acquisition_fun_grp = acquisition_root["labels/"
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
                f"labels/{segmentation_group_name}/0"
            ]

            (dataset_metadata,
             sampling_positions) = self._prepare_datasets_metadata(
                 image_group,
                 output_axes,
                 displayed_source_axes,
                 displayed_shape,
                 [(input_layers_group, "images"),
                  (sampling_mask_layers_group, "masks")]
                )

            # Compute acquisition function of the current image
            img_sampling_positions = self.compute_acquisition(
                dataset_metadata,
                acquisition_fun=acquisition_fun_grp,
                segmentation_out=segmentation_grp,
                sampling_positions=sampling_positions,
                segmentation_only=segmentation_only
            )

            self._update_image_progressbar(n + 1)

            if not img_sampling_positions:
                continue

            if not segmentation_only:
                add_multiscale_output_layer(
                    acquisition_root,
                    axes=output_axes,
                    scale=output_scale,
                    data_group="labels/acquisition_fun/0",
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

            segmentation_channel = add_multiscale_output_layer(
                segmentation_root,
                axes=output_axes,
                scale=output_scale,
                data_group=f"labels/{segmentation_group_name}/0",
                group_name=group_name + f" {segmentation_group_name}",
                layers_group_name=segmentation_group_name,
                image_group=image_group,
                reference_source_axes=displayed_source_axes,
                reference_scale=displayed_scale,
                output_filename=output_filename,
                add_func=viewer.add_labels
            )

            if not segmentation_only and image_group.labels_group is None:
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

        if (not image_groups
           or not self.labels_manager.labels_group_root.childCount()):
            return False

        dataset_metadata_list = []

        for image_group in image_groups:
            image_group.setSelected(True)

            input_layers_group_idx = image_group.input_layers_group

            segmentation_layers_group = image_group.getLayersGroup(
                layers_group_name="segmentation"
            )

            if (input_layers_group_idx is None
               or segmentation_layers_group is None):
                continue

            input_layers_group = image_group.child(input_layers_group_idx)

            displayed_source_axes = input_layers_group.source_axes
            displayed_shape = input_layers_group.shape

            output_axes = displayed_source_axes
            if "C" in output_axes:
                output_axes = list(output_axes)
                output_axes.remove("C")
                output_axes = "".join(output_axes)

            (dataset_metadata,
             sampling_positions) = self._prepare_datasets_metadata(
                 image_group,
                 output_axes,
                 displayed_source_axes,
                 displayed_shape,
                 [(input_layers_group, "images"),
                  (segmentation_layers_group, "labels")]
                )

            dataset_metadata_list.append((dataset_metadata,
                                          sampling_positions))

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
