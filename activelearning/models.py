from typing import Iterable
from cellpose import train, transforms, models
from torch.utils.data import DataLoader, ChainDataset
import random
import zarrdataset as zds


def train_cellpose(model, dataset_metadata_list: Iterable[dict],
                   train_data_proportion: float = 0.8,
                   patch_size: int = 256,
                   spatial_axes="ZYX",
                   **kwargs):

    datasets = [
        zds.ZarrDataset(
            list(dataset_metadata.values()),
            return_positions=False,
            draw_same_chunk=True,
            patch_sampler=zds.PatchSampler(
                patch_size=patch_size,
                spatial_axes=dataset_metadata["masks"]["axes"],
                min_area=1),
            shuffle=True
        )
        for dataset_metadata in dataset_metadata_list
    ]

    chained_datasets = ChainDataset(datasets)

    dataloader = DataLoader(
        chained_datasets,
        num_workers=kwargs.get("num_workers", 0),
        worker_init_fn=zds.chained_zarrdataset_worker_init_fn
    )

    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    for x, t in dataloader:
        lab = t[0].numpy()
        img = x[0].numpy()
        img_t = transforms.convert_image(img, channel_axis=2, channels=[0, 0])
        img_t = transforms.normalize_img(img_t, invert=False, axis=2)

        img_t = img_t.transpose(2, 0, 1)

        if random.random() <= train_data_proportion:
            train_data.append(img_t)
            train_labels.append(lab)
        else:
            test_data.append(img_t)
            test_labels.append(lab)

    if not test_data:
        # Take at least one sample at random from the train dataset
        test_data_idx = random.randrange(0, len(train_data))
        test_data = [train_data.pop(test_data_idx)]
        test_labels = [train_labels.pop(test_data_idx)]

    model_path = train.train_seg(
        model.net,
        train_data=train_data,
        train_labels=train_labels,
        train_probs=None,
        test_data=test_data,
        test_labels=test_labels,
        test_probs=None,
        load_files=True,
        **kwargs
    )

    model = models.CellposeModel(pretrained_model=model_path)

    return model
