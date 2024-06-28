import random
from cellpose import train
import zarrdataset as zds


def train_cellpose(model, dataset_metadata, train_data_proportion: float = 0.8,
                   patch_size: int = 256):
    dataset = zds.ZarrDataset(
        dataset_metadata,
        patch_sampler=zds.PatchSampler(patch_size=patch_size),
        draw_same_chunk=True
    )

    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    for x, t in dataset:
        if random.random() <= train_data_proportion:
            train_data.append(x)
            train_labels.append(t)
        else:
            test_data.append(x)
            test_labels.append(t)

    model_path = train.train_seg(
        model,
        train_data=train_data,
        train_labels=None,
        train_probs=None,
        test_data=test_data,
        test_labels=test_labels,
        test_probs=None,
        load_files=True,
        batch_size=8,
        learning_rate=0.005,
        n_epochs=2000,
        weight_decay=1e-5,
        momentum=0.9,
        SGD=False,
        channels=None,
        channel_axis=None,
        rgb=False,
        normalize=True,
        compute_flows=False,
        save_path=None,
        save_every=100,
        nimg_per_epoch=None,
        nimg_test_per_epoch=None,
        rescale=True,
        scale_range=None,
        bsize=224,
        min_train_masks=5,
        model_name=None
    )

    model = model.CellposeModel(model_path)

    return model
