from typing import Iterable, Union
from functools import partial

import numpy as np
import zarr
import zarrdataset as zds

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QWidget, QGridLayout, QScrollArea, QCheckBox,
                            QSpinBox,
                            QLabel)

from time import perf_counter

try:
    import torch
    from torch.utils.data import DataLoader, ChainDataset
    USING_PYTORCH = True
except ModuleNotFoundError:
    USING_PYTORCH = False


class SegmentationMethod:
    model_axes = ""

    def __init__(self):
        super().__init__()

    def _model_init(self):
        raise NotImplementedError("This method requies to be overriden by a "
                                  "derived class.")

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


class AxesCorrector:
    def __init__(self, out_axes: str, target_axes: str):
        self.out_axes = list(out_axes)
        self.target_axes = list(target_axes)
        self._drop_axes = list(set(out_axes) - set(target_axes))
        self.permute_order = zds.map_axes_order(out_axes, target_axes)

    def __call__(self, img):
        img_corr = img.transpose(self.permute_order)

        # Drop axes with length 1 that are not in `axes`.
        out_shape = [s
                     for s, p_a in zip(img_corr.shape, self.permute_order)
                     if self.out_axes[p_a] not in self._drop_axes]
        img_corr = img_corr.reshape(out_shape)
        return img_corr


class InvalidSample(Exception):
    """Exception raised when a sample is invalid due to lack of enugh labels or
    not big-enough label objects."""
    def __init__(self, message):
        super().__init__(message)

    def __str__(self):
        return f"InvalidSample: {self.args[0]}"


class MyZarrDataset(zds.ZarrDataset):
    def __init__(self, *args, max_samples=None, repetitions_per_sample=1,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.repetitions_per_sample = repetitions_per_sample
        self.max_samples = max_samples

    def _estimate_dataset_size(self):
        if self.max_samples is not None:
            return

        self.max_samples = 0
        for im_coll, tl_chks in zip(self._arr_lists, self._toplefts):
            for chunk_tlbr in tl_chks:
                patches_tls = self._patch_sampler.compute_patches(
                    im_coll,
                    chunk_tlbr
                )

                self.max_samples += len(patches_tls)

    def __len__(self):
        self._estimate_dataset_size()
        return self.max_samples

    def __getstate__(self):
        # Custom behavior for pickling the ZarrDataset object.
        state = self.__dict__.copy()
        # Remove any attributes that should not be pickled
        state['_arr_lists'] = None
        state['_curr_collection'] = None
        state['_initialized'] = False

        return state

    def __setstate__(self, state):
        # Custom behavior for unpickling the ZarrDataset object.
        self.__dict__.update(state)

    def _initialize(self, force=False):
        if self._initialized and not force:
            return

        super()._initialize(force=force)

        if self.repetitions_per_sample > 1:
            self._arr_lists = np.tile(self._arr_lists,
                                      (self.repetitions_per_sample, ))

            self._toplefts = np.tile(self._toplefts,
                                     (self.repetitions_per_sample, 1))

    def __iter__(self):
        self._initialize()

        n_samples = 0

        self._estimate_dataset_size()

        max_samples = self.max_samples // self._num_workers
        remaining_samples = self.max_samples % self._num_workers
        if self._worker_id < remaining_samples:
            max_samples += 1

        while n_samples < max_samples:
            iter = super().__iter__()

            while n_samples < max_samples:
                try:
                    batch = next(iter)

                    yield batch
                    n_samples += 1

                except InvalidSample:
                    continue

                except StopIteration:
                    # If there are no more valid samples in this shard of the
                    # dataset, terminate the loop
                    break
            else:
                # This means we have extracted the maximum amount of samples
                # requested
                break

            if n_samples == 0:
                # If we have not extracted any samples, we need to break the
                # loop to avoid an infinite loop
                break


class TunableMethod(SegmentationMethod):
    def __init__(self):
        self.max_samples = 0
        self.repetitions_per_sample = 1

        super().__init__()

    def get_train_transform(self, *args, **kwargs) -> dict:
        raise NotImplementedError("This method requies to be overriden by a "
                                  "derived class.")

    def get_inference_transform(self, *args, **kwargs) -> dict:
        raise NotImplementedError("This method requies to be overriden by a "
                                  "derived class.")

    def _fine_tune(self, train_dataloader, val_dataloader) -> bool:
        raise NotImplementedError("This method requies to be overriden by a "
                                  "derived class.")

    def fine_tune(self, dataset_metadata_list: Iterable[dict],
                  model_axes: str,
                  train_data_proportion: float = 0.8,
                  patch_sizes: Union[dict, int] = 256,
                  num_workers: int = 0) -> bool:

        base_mode_transforms = self.get_train_transform()

        if base_mode_transforms is None:
            base_mode_transforms = {}

        base_mode_transforms = {
            input_mode: [mode_transforms]
            for input_mode, mode_transforms in
            base_mode_transforms.items()
        }

        # Complete the transforms for individial input modes
        mode_transforms = {
            (input_mode, ): []
            for input_mode in dataset_metadata_list[0].keys()
            if (input_mode, ) not in base_mode_transforms
        }
        mode_transforms.update(base_mode_transforms)

        for input_mode in dataset_metadata_list[0].keys():
            mode_transforms[(input_mode, )].insert(0, AxesCorrector(
                dataset_metadata_list[0][input_mode]["axes"],
                model_axes
            ))

        worker_init_fn = None

        if len(dataset_metadata_list) == 1:
            if isinstance(dataset_metadata_list[0]["masks"]["filenames"], str):
                z_grp = zarr.open(
                    dataset_metadata_list[0]["masks"]["filenames"],
                    mode="r"
                )

                sampling_mask = np.copy(
                    z_grp[dataset_metadata_list[0]["masks"]["data_group"]][:]
                )
            elif isinstance(dataset_metadata_list[0]["masks"]["filenames"],
                            np.ndarray):
                sampling_mask = np.copy(
                    dataset_metadata_list[0]["masks"]["filenames"]
                )
            else:
                raise ValueError("The mask filenames must be a numpy array or "
                                 "a Zarr file.")

            sampling_locations = np.nonzero(sampling_mask)
            sampling_locations = np.ravel_multi_index(sampling_locations,
                                                      sampling_mask.shape)

            trn_samples = int(len(sampling_locations) * train_data_proportion)
            sampling_locations = np.random.choice(
                sampling_locations,
                size=trn_samples,
                replace=False
            )
            sampling_locations = np.unravel_index(sampling_locations,
                                                  sampling_mask.shape)

            train_mask = np.zeros_like(sampling_mask)
            train_mask[sampling_locations] = True
            val_mask = np.bitwise_xor(train_mask, sampling_mask)

            patch_sampler = zds.PatchSampler(
                patch_size=patch_sizes,
                spatial_axes=dataset_metadata_list[0]["labels"]["axes"],
                min_area=1
            )

            dataset_metadata_list[0]["masks"]["filenames"] = train_mask

            train_datasets = MyZarrDataset(
                list(dataset_metadata_list[0].values()),
                return_positions=False,
                draw_same_chunk=False,
                patch_sampler=patch_sampler,
                shuffle=True,
                repetitions_per_sample=self.repetitions_per_sample,
                max_samples=self.max_samples if self.max_samples > 0 else None
            )

            dataset_metadata_list[0]["masks"]["filenames"] = val_mask

            val_datasets = MyZarrDataset(
                list(dataset_metadata_list[0].values()),
                return_positions=False,
                draw_same_chunk=False,
                patch_sampler=patch_sampler,
                shuffle=True,
                repetitions_per_sample=self.repetitions_per_sample,
                max_samples=self.max_samples if self.max_samples > 0 else None
            )

            for input_mode, transform_mode in mode_transforms.items():
                train_datasets.add_transform(input_mode, transform_mode)

            for input_mode, transform_mode in mode_transforms.items():
                val_datasets.add_transform(input_mode, transform_mode)

            worker_init_fn = zds.zarrdataset_worker_init_fn

        else:
            train_datasets = []
            val_datasets = []

            training_indices = np.random.choice(
                len(dataset_metadata_list),
                int(train_data_proportion * len(dataset_metadata_list))
            ).tolist()

            for idx, dataset_metadata in enumerate(dataset_metadata_list):
                patch_sampler = zds.PatchSampler(
                    patch_size=patch_sizes,
                    spatial_axes=dataset_metadata["labels"]["axes"],
                    min_area=0.01
                )

                dataset = MyZarrDataset(
                    list(dataset_metadata.values()),
                    return_positions=False,
                    draw_same_chunk=False,
                    patch_sampler=patch_sampler,
                    shuffle=True,
                    repetitions_per_sample=self.repetitions_per_sample,
                    max_samples=(
                        self.max_samples if self.max_samples > 0 else None
                    )
                )

                for input_mode, transform_mode in mode_transforms.items():
                    dataset.add_transform(input_mode, transform_mode)

                if idx in training_indices:
                    train_datasets.append(dataset)
                else:
                    val_datasets.append(dataset)

            train_datasets = ChainDataset(train_datasets)
            val_datasets = ChainDataset(val_datasets)
            worker_init_fn = zds.chained_zarrdataset_worker_init_fn

        if USING_PYTORCH:
            train_dataloader = DataLoader(
                train_datasets,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn
            )
            val_dataloader = DataLoader(
                val_datasets,
                num_workers=num_workers,
                worker_init_fn=worker_init_fn
            )
        else:
            train_dataloader = train_datasets
            val_dataloader = val_datasets

        return self._fine_tune(train_dataloader, val_dataloader)


class TunableWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.refresh_model = True

        (self._segmentation_parameters,
         segmentation_parameter_names) = self._segmentation_parameters_widget()

        (self._finetuning_parameters,
         finetuning_parameter_names) = self._finetuning_parameters_widget()

        self.advanced_segmentation_options_chk = QCheckBox(
            "Advanced segmentation parameters"
        )

        self.advanced_segmentation_options_chk.setChecked(False)
        self.advanced_segmentation_options_chk.toggled.connect(
            self._show_segmentation_parameters
        )

        self.advanced_finetuning_options_chk = QCheckBox(
            "Advanced fine tuning parameters"
        )

        self.advanced_finetuning_options_chk.setChecked(False)
        self.advanced_finetuning_options_chk.toggled.connect(
            self._show_finetuning_parameters
        )

        self._segmentation_parameters_scr = QScrollArea()
        self._segmentation_parameters_scr.setWidgetResizable(True)
        self._segmentation_parameters_scr.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )

        self._finetuning_parameters_scr = QScrollArea()
        self._finetuning_parameters_scr.setWidgetResizable(True)
        self._finetuning_parameters_scr.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )

        if self._segmentation_parameters is not None:
            self._segmentation_parameters_scr.setWidget(
                self._segmentation_parameters.native
            )

            for par_name in segmentation_parameter_names:
                self._segmentation_parameters.__getattr__(par_name)\
                                                .changed.connect(
                    partial(self._set_parameter, parameter_key="_" + par_name)
                )

        if self._finetuning_parameters is not None:
            self._finetuning_parameters_scr.setWidget(
                self._finetuning_parameters.native
            )

            for par_name in finetuning_parameter_names:
                self._finetuning_parameters.__getattr__(par_name).changed\
                                                                    .connect(
                    partial(self._set_parameter, parameter_key="_" + par_name)
                )

        self.max_samples_spn = QSpinBox(
            minimum=0,
            maximum=10000,
            value=0,
            singleStep=1
        )
        self.repetitions_per_sample_spn = QSpinBox(
            minimum=1,
            maximum=100,
            value=1,
            singleStep=1
        )

        self.parameters_lyt = QGridLayout()
        self.parameters_lyt.addWidget(
            self.advanced_segmentation_options_chk, 0, 0
        )
        self.parameters_lyt.addWidget(
            self.advanced_finetuning_options_chk, 2, 0
        )
        self.parameters_lyt.addWidget(
            self._segmentation_parameters_scr, 1, 0, 1, 2
        )
        self.parameters_lyt.addWidget(
            self._finetuning_parameters_scr, 3, 0, 1, 2
        )

        self.parameters_lyt.addWidget(QLabel("Number of samples per epoch "
                                             "(fine tuning only):"), 4, 0)
        self.parameters_lyt.addWidget(self.max_samples_spn, 4, 1)
        self.parameters_lyt.addWidget(QLabel("Repetitions per sample "
                                             "(fine tuning only):"), 5, 0)
        self.parameters_lyt.addWidget(self.repetitions_per_sample_spn, 5, 1)

        self.setLayout(self.parameters_lyt)

        self.max_samples_spn.valueChanged.connect(
            self._set_max_samples
        )

        self.repetitions_per_sample_spn.valueChanged.connect(
            self._set_repetitions_per_sample
        )

        self._segmentation_parameters_scr.hide()
        self._finetuning_parameters_scr.hide()

    def _segmentation_parameters_widget(self):
        raise NotImplementedError("This method requies to be overriden by a "
                                  "derived class.")

    def _finetuning_parameters_widget(self):
        raise NotImplementedError("This method requies to be overriden by a "
                                  "derived class.")

    def _check_parameters(self, parameter_val, parameter_key=None):
        return parameter_val

    def _set_parameter(self, parameter_val, parameter_key=None):
        parameter_val = self._check_parameter(parameter_val, parameter_key)

        if (getattr(self, parameter_key) is None and parameter_val is not None
           or getattr(self, parameter_key) != parameter_val):
            self.refresh_model = True
            setattr(self, parameter_key, parameter_val)

    def _set_max_samples(self, max_samples: int):
        if max_samples > 0 and max_samples != self.max_samples:
            self.max_samples = max_samples

    def _set_repetitions_per_sample(self, repetitions_per_sample: int):
        if repetitions_per_sample != self.repetitions_per_sample:
            self.repetitions_per_sample = repetitions_per_sample

    def _show_segmentation_parameters(self, show: bool):
        self._segmentation_parameters_scr.setVisible(show)

    def _show_finetuning_parameters(self, show: bool):
        self._finetuning_parameters_scr.setVisible(show)
