from typing import Iterable, Union
import numpy as np
import zarrdataset as zds

try:
    import torch
    from torch.utils.data import DataLoader, ChainDataset
    USING_PYTORCH = True
except ModuleNotFoundError:
    USING_PYTORCH = False


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


class MyZarrDataset(zds.ZarrDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_samples_per_image = 0

    def __len__(self):
        return self.max_samples_per_image * len(self._collections["images"])

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


class TunableMethod(SegmentationMethod):
    def __init__(self):
        self._num_workers = 0
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
                  train_data_proportion: float = 0.8,
                  patch_sizes: Union[dict, int] = 256):

        mode_transforms = self.get_train_transform()

        worker_init_fn = None

        if len(dataset_metadata_list) == 1:
            sampling_mask = np.copy(
                dataset_metadata_list[0]["masks"]["filenames"]
            )

            sampling_locations = np.nonzero(sampling_mask)
            sampling_locations = np.ravel_multi_index(sampling_locations,
                                                      sampling_mask.shape)

            trn_samples = int(train_data_proportion * len(sampling_locations))
            val_samples = len(sampling_locations) - trn_samples

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
                min_area=0.01
            )

            dataset_metadata_list[0]["masks"]["filenames"] = train_mask

            train_datasets = MyZarrDataset(
                list(dataset_metadata_list[0].values()),
                return_positions=False,
                draw_same_chunk=False,
                patch_sampler=patch_sampler,
                shuffle=True,
            )
            train_datasets.max_samples_per_image = trn_samples

            dataset_metadata_list[0]["masks"]["filenames"] = val_mask

            val_datasets = MyZarrDataset(
                list(dataset_metadata_list[0].values()),
                return_positions=False,
                draw_same_chunk=False,
                patch_sampler=patch_sampler,
                shuffle=True,
            )

            val_datasets.max_samples_per_image = val_samples

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
                )

                dataset.max_samples_per_image = self.max_samples_per_image
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
                num_workers=1,
                worker_init_fn=worker_init_fn
            )
            val_dataloader = DataLoader(
                val_datasets,
                num_workers=1,
                worker_init_fn=worker_init_fn
            )
        else:
            train_dataloader = train_datasets
            val_dataloader = val_datasets

        return self._fine_tune(train_dataloader, val_dataloader)
