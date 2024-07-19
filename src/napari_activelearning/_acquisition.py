from typing import Optional, Iterable, Tuple
import random
import numpy as np

import zarrdataset as zds
import dask.array as da

try:
    import torch
    from torch.utils.data import DataLoader, ChainDataset
    USING_TORCH = True
except ModuleNotFoundError:
    from itertools import chain
    USING_TORCH = False

import napari
from napari.layers._multiscale_data import MultiScaleData

from ._layers import ImageGroupsManager, ImageGroup
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


if USING_TORCH:
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
        self._transform = None

        super().__init__()

    def _fine_tune(self, train_data, train_labels, test_data, test_labels):
        raise NotImplementedError("This method requies to be overriden by a "
                                  "derived class.")

    def fine_tune(self, dataset_metadata_list: Iterable[
        Tuple[dict, Iterable[Iterable[int]]]],
                  train_data_proportion: float = 0.8,
                  patch_size: int = 256,
                  spatial_axes="ZYX"):
        # Add the pre-processing transform function to the metadata
        # dictionary at "images" as "image_loader_func".
        for dataset_metadata in dataset_metadata_list:
            dataset_metadata[0]["images"]["image_loader_func"] =\
                self._transform

        datasets = [
            zds.ZarrDataset(
                list(dataset_metadata.values()),
                return_positions=False,
                draw_same_chunk=True,
                patch_sampler=StaticPatchSampler(
                    patch_size=patch_size,
                    top_lefts=top_lefts,
                    spatial_axes=spatial_axes
                ),
                shuffle=True,
            )
            for dataset_metadata, top_lefts in dataset_metadata_list
        ]

        chained_datasets = ChainDataset(datasets)

        if USING_TORCH:
            dataloader = DataLoader(
                chained_datasets,
                num_workers=self._num_workers,
                worker_init_fn=zds.chained_zarrdataset_worker_init_fn
            )
        else:
            dataloader = chain(datasets)

        train_data = []
        test_data = []
        train_labels = []
        test_labels = []

        for img, lab in dataloader:
            if USING_TORCH:
                if isinstance(img, torch.Tensor):
                    img = img.numpy()

                if isinstance(lab, torch.Tensor):
                    lab = lab.numpy()

                img = img[0]
                lab = lab[0]

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


class TunableMethod(SegmentationMethod, FineTuningMethod):
    def __init__(self):
        super().__init__()


class AcquisitionFunction:
    def __init__(self, image_groups_manager: ImageGroupsManager,
                 labels_manager: LabelsManager,
                 tunable_segmentation_method: TunableMethod):
        self._patch_size = 128
        self._max_samples = 1
        self._MC_repetitions = 3

        viewer = napari.current_viewer()
        self._input_axes = "".join(viewer.dims.axis_labels).upper()

        self.image_groups_manager = image_groups_manager
        self.labels_manager = labels_manager
        self.tunable_segmentation_method = tunable_segmentation_method

        super().__init__()

    def _update_roi_from_position(self):
        viewer = napari.current_viewer()
        viewer_axes = "".join(viewer.dims.axis_labels).upper()
        position = viewer.cursor.position
        axes_order = viewer.dims.order

        self._roi = {
            viewer_axes[ord]:
            slice(None)
            if viewer_axes[ord] in self._input_axes
            else slice(int(position[ord]), int(position[ord]) + 1)
            for ord in axes_order
        }

    def _compute_acquisition_fun(self, img, img_sp, MC_repetitions):
        probs = []
        for _ in range(MC_repetitions):
            probs.append(
                self.tunable_segmentation_method.probs(img)
            )
        probs = np.stack(probs, axis=0)

        u_sp_lab = compute_acquisition_superpixel(probs, img_sp)

        return u_sp_lab

    def _compute_segmentation(self, img, labels_offset=0):
        seg_out = self.tunable_segmentation_method.segment(img)
        seg_out = np.where(seg_out, seg_out + labels_offset, 0)
        return seg_out

    def _reset_image_progressbar(self, num_images: int):
        pass

    def _update_image_progressbar(self, curr_image_index: int):
        pass

    def _reset_patch_progressbar(self):
        pass

    def _update_patch_progressbar(self, curr_patch_index: int):
        pass

    def compute_acquisition(self, dataset_metadata, acquisition_fun,
                            segmentation_out,
                            sampling_positions=None,
                            segmentation_only=False,
                            spatial_axes="ZYX"):
        dl = get_dataloader(dataset_metadata, patch_size=self._patch_size,
                            sampling_positions=sampling_positions,
                            spatial_axes=spatial_axes,
                            shuffle=True)
        segmentation_max = 0
        n_samples = 0
        img_sampling_positions = []

        if "masks" in dataset_metadata:
            mask_axes = dataset_metadata["masks"]["source_axes"]
        else:
            mask_axes = spatial_axes

        pred_sel = tuple(
            slice(None) if ax in pred_spatial_axes else None
            for ax in mask_axes
        )

        self._reset_patch_progressbar()
        for pos, img, img_sp in dl:
            if USING_TORCH:
                pos = pos[0].numpy()
                img = img[0].numpy()
                img_sp = img_sp.numpy().squeeze()
            else:
                img_sp = img_sp.squeeze()

            pos = {
                ax: slice(pos_ax[0], pos_ax[1])
                for ax, pos_ax in zip(dataset_metadata["images"]["axes"],
                                      pos)
            }

            pos_u_lab = tuple(
                pos.get(ax, self._roi[ax])
                for ax in mask_axes
            )

            if not segmentation_only:
                u_sp_lab = self._compute_acquisition_fun(
                    img,
                    img_sp,
                    self._MC_repetitions,
                )
                acquisition_fun[pos_u_lab] = u_sp_lab[pred_sel]
                acquisition_val = u_sp_lab.max()
            else:
                acquisition_val = 0

            seg_out = self._compute_segmentation(
                img,
                segmentation_max
            )
            segmentation_out[pos_u_lab] = seg_out[pred_sel]
            segmentation_max = max(segmentation_max, seg_out.max())

            img_sampling_positions.append(
                LabelItem(acquisition_val, position=pos_u_lab)
            )

            n_samples += 1
            self.patch_pb.setValue(n_samples)
            if n_samples >= self._max_samples:
                break

            self._update_patch_progressbar(n_samples)

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
            self.image_groups_manager.image_groups_tw.selectedItems()
        ))

        if not image_groups:
            return

        self._update_roi_from_position()

        self._reset_image_progressbar(len(image_groups))

        viewer = napari.current_viewer()
        spatial_axes = self._input_axes
        if "C" in spatial_axes:
            spatial_axes = list(spatial_axes)
            spatial_axes.remove("C")
            spatial_axes = "".join(spatial_axes)

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

            acquisition_fun_shape, acquisition_fun_scale = list(zip(*[
                (ax_s, ax_scl)
                for ax, ax_s, ax_scl in zip(displayed_source_axes,
                                            displayed_shape,
                                            displayed_scale)
                if ax in spatial_axes and ax_s > 1
            ]))

            if not segmentation_only:
                acquisition_root = save_zarr(
                    output_filename,
                    data=None,
                    shape=acquisition_fun_shape,
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
                shape=acquisition_fun_shape,
                chunk_size=True,
                name=segmentation_group_name,
                dtype=np.int32,
                is_label=True,
                is_multiscale=True
            )

            segmentation_grp = segmentation_root[
                f"labels/{segmentation_group_name}/0"
            ]

            dataset_metadata = {}

            for layers_group, layer_type in [
               (input_layers_group, "images"),
               (sampling_mask_layers_group, "masks")
               ]:
                if layers_group is None:
                    continue

                dataset_metadata[layer_type] = layers_group.metadata
                dataset_metadata[layer_type]["roi"] = [
                        self._roi.get(ax, slice(None))
                        for ax in layers_group.source_axes
                ]

                if "images" in layer_type:
                    for ax in spatial_axes:
                        if ax not in displayed_source_axes:
                            continue

                        ax_idx = displayed_source_axes.index(ax)

                        if displayed_shape[ax_idx] < self._patch_size:
                            continue

                        dataset_metadata[layer_type]["roi"][ax_idx] = slice(
                            0,
                            self._patch_size * (displayed_shape[ax_idx]
                                                // self._patch_size)
                        )

                dataset_metadata[layer_type]["roi"] = [tuple(
                    dataset_metadata[layer_type]["roi"]
                )]

                if isinstance(dataset_metadata[layer_type]["filenames"],
                              MultiScaleData):
                    dataset_metadata[layer_type]["filenames"] =\
                        dataset_metadata[layer_type]["filenames"][0]

                dataset_metadata[layer_type]["modality"] = layer_type

                if "images" in layer_type:
                    dataset_metadata[layer_type]["axes"] = self._input_axes
                else:
                    dataset_metadata[layer_type]["axes"] = spatial_axes

            if image_group.labels_group:
                sampling_positions = list(
                    map(lambda child:
                        [ax_pos.start for ax_pos in child.position],
                        map(lambda idx: image_group.labels_group.child(idx),
                            range(image_group.labels_group.childCount())))
                )
            else:
                sampling_positions = None

            # Compute acquisition function of the current image
            img_sampling_positions = self.compute_acquisition(
                dataset_metadata,
                acquisition_fun=acquisition_fun_grp,
                segmentation_out=segmentation_grp,
                sampling_positions=sampling_positions,
                segmentation_only=segmentation_only,
                spatial_axes=spatial_axes
            )

            self._update_image_progressbar(n + 1)

            if not img_sampling_positions:
                continue

            if not segmentation_only:
                if output_filename:
                    acquisition_root = output_filename

                # Downsample the acquisition function
                acquisition_fun_ms = downsample_image(
                    acquisition_root,
                    source_axes=spatial_axes,
                    data_group="labels/acquisition_fun/0",
                    scale=4,
                    num_scales=5,
                    reference_source_axes=displayed_source_axes,
                    reference_scale=displayed_scale
                )

                new_acquisition_layer = viewer.add_image(
                    acquisition_fun_ms,
                    name=group_name + " acquisition function",
                    multiscale=True,
                    opacity=0.8,
                    scale=acquisition_fun_scale,
                    blending="translucent_no_depth",
                    colormap="magma",
                    contrast_limits=(
                        0,
                        max(img_sampling_positions).acquisition_val
                    ),
                )

                if isinstance(new_acquisition_layer, list):
                    new_acquisition_layer = new_acquisition_layer[0]

                acquisition_layers_group = image_group.getLayersGroup(
                    "acquisition"
                )

                if acquisition_layers_group is None:
                    acquisition_layers_group = image_group.add_layers_group(
                        "acquisition",
                        source_axes=spatial_axes,
                        use_as_input_image=False,
                        use_as_sampling_mask=False
                    )

                acquisition_channel = acquisition_layers_group.add_layer(
                    new_acquisition_layer
                )

                if output_filename:
                    acquisition_channel.source_data = str(output_filename)
                    acquisition_channel.data_group = "labels/acquisition_fun/0"

            if output_filename:
                segmentation_root = output_filename

            # Downsample the segmentation output
            segmentation_ms = downsample_image(
                segmentation_root,
                source_axes=spatial_axes,
                data_group=f"labels/{segmentation_group_name}/0",
                scale=4,
                num_scales=5,
                reference_source_axes=displayed_source_axes,
                reference_scale=displayed_scale,
            )

            new_segmentation_layer = viewer.add_labels(
                segmentation_ms,
                name=group_name + f" {segmentation_group_name}",
                multiscale=True,
                opacity=0.8,
                scale=acquisition_fun_scale,
                blending="translucent_no_depth"
            )

            if isinstance(new_segmentation_layer, list):
                new_segmentation_layer = new_segmentation_layer[0]

            segmentation_layers_group = image_group.getLayersGroup(
                segmentation_group_name
            )

            if segmentation_layers_group is None:
                segmentation_layers_group = image_group.add_layers_group(
                    segmentation_group_name,
                    source_axes=spatial_axes,
                    use_as_input_image=False,
                    use_as_sampling_mask=False
                )

            segmentation_channel = segmentation_layers_group.add_layer(
                new_segmentation_layer
            )

            if not segmentation_only and image_group.labels_group is None:
                new_label_group = self.labels_manager.add_labels(
                    segmentation_channel,
                    img_sampling_positions
                )

                image_group.labels_group = new_label_group

            if output_filename:
                segmentation_channel.source_data = str(output_filename)
                segmentation_channel.data_group =\
                    f"labels/{segmentation_group_name}/0"

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
            return

        patch_size = self.patch_size_spn.value()

        dataset_metadata_list = []

        spatial_axes = self._input_axes
        if "C" in spatial_axes:
            spatial_axes = list(spatial_axes)
            spatial_axes.remove("C")
            spatial_axes = "".join(spatial_axes)

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

            dataset_metadata = {}

            for layers_group, layer_type in [
               (input_layers_group, "images"),
               (segmentation_layers_group, "labels")
               ]:
                dataset_metadata[layer_type] = layers_group.metadata
                dataset_metadata[layer_type]["roi"] = [
                    tuple(
                        self._roi.get(ax, slice(None))
                        for ax in layers_group.source_axes
                    )
                ]

                if isinstance(dataset_metadata[layer_type]["filenames"],
                              MultiScaleData):
                    dataset_metadata[layer_type]["filenames"] =\
                        dataset_metadata[layer_type]["filenames"][0]

                if isinstance(dataset_metadata[layer_type]["filenames"],
                              da.core.Array):
                    dataset_metadata[layer_type]["filenames"] =\
                        dataset_metadata[layer_type]["filenames"].compute()

                dataset_metadata[layer_type]["modality"] = layer_type

                if "images" in layer_type:
                    dataset_metadata[layer_type]["axes"] = self._input_axes
                else:
                    dataset_metadata[layer_type]["axes"] = spatial_axes

            sampling_positions = list(
                map(lambda child: [ax_pos.start for ax_pos in child.position],
                    map(lambda idx: image_group.labels_group.child(idx),
                        range(image_group.labels_group.childCount())))
            )

            dataset_metadata_list.append((dataset_metadata,
                                          sampling_positions))

        self.tunable_segmentation_method.fine_tune(dataset_metadata_list,
                                                   patch_size=patch_size,
                                                   spatial_axes=spatial_axes)

        self.compute_acquisition_layers(
            run_all=True,
            segmentation_group_name="fine_tunned_segmentation",
            segmentation_only=True
        )
