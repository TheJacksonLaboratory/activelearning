import numpy as np
import skimage

import zarrdataset as zds

from ._acquisition import add_dropout, USING_PYTORCH
from ._models import TunableMethod, InvalidSample

try:
    import cellpose
    from cellpose import core, transforms, models, train
    import torch

    class CellposeTransform(zds.MaskGenerator):
        def __init__(self, channels=None, channel_axis=None):
            self._channel_axis = channel_axis
            self._channels = channels
            axes = ["Y", "X"]

            if self._channel_axis is not None:
                axes.insert(self._channel_axis, "C")

            axes = "".join(axes)

            super(CellposeTransform, self).__init__(axes=axes)

        def _compute_transform(self, image: np.ndarray) -> np.ndarray:
            img_t = transforms.convert_image(image,
                                             channel_axis=self._channel_axis,
                                             channels=self._channels)
            img_t = transforms.normalize_img(img_t, invert=False,
                                             axis=self._channel_axis)
            return img_t

    class EnsureInputs:
        def __init__(self, min_labels_per_sample=1, min_labels_size=1):
            self._min_labels_per_sample = min_labels_per_sample
            self._min_labels_size = min_labels_size

        def __call__(self, inputs, labels):
            if isinstance(labels, np.ndarray):
                labels_flat = labels.flatten()
            elif isinstance(labels, torch.Tensor):
                labels_flat = labels.flatten().cpu().numpy()

            unique_samples = set(np.unique(labels_flat)) - {0}
            if len(unique_samples) < self._min_labels_per_sample:
                raise InvalidSample(
                    f"Sample has {len(unique_samples)} labels, which is less "
                    f"than the required {self._min_labels_per_sample} labels")

            for new_label, label in enumerate(unique_samples, 1):
                labels_flat = np.where(labels_flat == label,
                                       new_label,
                                       labels_flat)

            labels_flat_count = np.bincount(labels_flat)
            if labels_flat.min() == 0:
                labels_size = labels_flat_count[1:].min()
            else:
                labels_size = labels_flat_count.min()

            if labels_size < self._min_labels_size:
                raise InvalidSample(
                    f"Sample's smaller label has size of {labels_size.min()} "
                    f"pixels/elements, which is less than the required minimum"
                    f" of {self._min_labels_size} pixels/elements")

            return inputs, labels

    class CellposeTunable(TunableMethod):
        model_axes = "YXC"
        _channel_axis = 2

        def __init__(self):
            super().__init__()

            self._custom_transform = None

            self._model = None
            self._model_dropout = None

            self.refresh_model = True

            self._pretrained_model = None
            self._model_type = "cyto"
            self._gpu = True
            self._channels = [0, 0]

            self._batch_size = 8
            self._learning_rate = 0.005
            self._n_epochs = 20
            self._weight_decay = 1e-5
            self._momentum = 0.9
            self._SGD = False
            self._rgb = False
            self._normalize = True
            self._compute_flows = False
            self._save_path = None
            self._save_every = 100
            self._nimg_per_epoch = None
            self._nimg_test_per_epoch = None
            self._rescale = True
            self._scale_range = None
            self._bsize = 224
            self._min_train_masks = 5
            self._model_name = None

        def _model_init(self):
            if not self.refresh_model:
                return

            gpu = torch.cuda.is_available() and self._gpu
            if self._pretrained_model is None:
                model_type = self._model_type
            else:
                model_type = None

            self._model = models.CellposeModel(
                gpu=gpu,
                model_type=model_type,
                pretrained_model=(str(self._pretrained_model)
                                  if self._pretrained_model is not None
                                  else None)
            )
            self._model.mkldnn = False
            self._model.net.mkldnn = False

            self._model_dropout = models.CellposeModel(
                gpu=gpu,
                model_type=model_type,
                pretrained_model=(str(self._pretrained_model)
                                  if self._pretrained_model is not None
                                  else None)
            )
            self._model_dropout.mkldnn = False
            self._model_dropout.net.mkldnn = False

            self._model_dropout.net.load_model(
                self._model_dropout.pretrained_model,
                device=self._model_dropout.device
            )
            add_dropout(self._model_dropout.net)
            self._model_dropout.net.eval()

            self._custom_transform = CellposeTransform(self._channels,
                                                       self._channel_axis)

            self.refresh_model = False

        def _run_pred(self, img, *args, **kwargs):
            self._model_init()

            with torch.no_grad():
                img_t = self._custom_transform(img)

                img_t = img_t[None, ...]

                if img_t.ndim < 4:
                    img_t = img_t[..., None]

                if img_t.shape[-1] == 1:
                    img_t = np.repeat(img_t, 2, axis=-1)

                y, _ = core.run_net(self._model_dropout.net, img_t)

                logits = torch.from_numpy(y[0, :, :, 2])
                probs = logits.sigmoid().numpy()

            return probs

        def _run_eval(self, img, *args, **kwargs):
            self._model_init()

            img_t = self._custom_transform(img)

            img_t = img_t[None, ...]

            if img_t.ndim < 4:
                img_t = img_t[..., None]

            if img_t.shape[-1] == 1:
                img_t = np.repeat(img_t, 2, axis=-1)

            seg, _, _ = self._model.eval(img_t, diameter=None,
                                         flow_threshold=None,
                                         channels=self._channels)
            return seg

        def get_train_transform(self, *args, **kwargs):
            mode_transforms = {
                ("images", "labels"): EnsureInputs()
            }
            return mode_transforms

        def get_inference_transform(self, *args, **kwargs):
            return None

        def _preload_data(self, dataloader):
            raw_data = []
            label_data = []
            for img, lab in dataloader:
                if USING_PYTORCH:
                    img = img[0].numpy().squeeze()
                    lab = lab[0].numpy().squeeze()

                    raw_data.append(img)
                    label_data.append(lab)

            return raw_data, label_data

        def _fine_tune(self, train_dataloader, val_dataloader) -> bool:
            self._model_init()

            (train_data,
             train_labels) = self._preload_data(
                 train_dataloader
             )

            (test_data,
             test_labels) = self._preload_data(
                 val_dataloader
             )

            self._pretrained_model = train.train_seg(
                self._model.net,
                train_data=train_data,
                train_labels=train_labels,
                train_probs=None,
                test_data=test_data,
                test_labels=test_labels,
                test_probs=None,
                load_files=False,
                batch_size=self._batch_size,
                learning_rate=self._learning_rate,
                n_epochs=self._n_epochs,
                weight_decay=self._weight_decay,
                momentum=self._momentum,
                SGD=self._SGD,
                channels=self._channels,
                channel_axis=self._channel_axis,
                rgb=self._rgb,
                normalize=self._normalize,
                compute_flows=self._compute_flows,
                save_path=self._save_path,
                save_every=self._save_every,
                nimg_per_epoch=self._nimg_per_epoch,
                nimg_test_per_epoch=self._nimg_test_per_epoch,
                rescale=self._rescale,
                scale_range=self._scale_range,
                bsize=self._bsize,
                min_train_masks=self._min_train_masks,
                model_name=self._model_name
            )

            if isinstance(self._pretrained_model, tuple):
                self._pretrained_model = self._pretrained_model[0]

            self.refresh_model = True

            return True

    USING_CELLPOSE = True

except ModuleNotFoundError:
    USING_CELLPOSE = False


class BaseTransform(zds.MaskGenerator):
    def __init__(self, channel_axis=None):
        self._channel_axis = channel_axis

        axes = ["Y", "X"]

        if self._channel_axis is not None:
            axes.insert(self._channel_axis, "C")

        axes = "".join(axes)

        super(BaseTransform, self).__init__(axes=axes)

    def _compute_transform(self, image: np.ndarray) -> np.ndarray:
        if "C" in self.axes:
            image_t = image.mean(axis=self.axes.index("C"))
        else:
            image_t = image

        return image_t


class SimpleTunable(TunableMethod):
    model_axes = "YXC"

    def __init__(self):
        super().__init__()
        self._channel_axis = 2
        self._threshold = 0.5

    def _model_init(self):
        pass

    def _run_pred(self, img, *args, **kwargs):
        img = img.astype(np.float32) + 1e-3 * np.random.randn(*img.shape)
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
        return img

    def _run_eval(self, img, *args, **kwargs):
        self._model_init()

        labels = skimage.measure.label(img > self._threshold)
        return labels

    def get_train_transform(self, *args, **kwargs):
        mode_transforms = {
            ("images", ): BaseTransform(channel_axis=self._channel_axis)
        }

        return mode_transforms

    def get_inference_transform(self, *args, **kwargs):
        mode_transforms = {
            ("images", ): BaseTransform(channel_axis=self._channel_axis)
        }

        return mode_transforms

    def _fine_tune(self, train_dataloader, val_dataloader) -> bool:
        return True
