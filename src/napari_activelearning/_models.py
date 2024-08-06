import os
import numpy as np
import skimage

import zarrdataset as zds

from ._acquisition import SegmentationMethod, FineTuningMethod, add_dropout


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

    class CellposeSegmentation(SegmentationMethod):
        def __init__(self):
            super().__init__()

            self._model = None
            self._model_dropout = None

            self.refresh_model = True

            self._transform = None

            self._pretrained_model = None
            self._model_type = "cyto"
            self._gpu = True
            self._channel_axis = 2
            self._channels = [0, 0]

        def _model_init(self):
            gpu = torch.cuda.is_available() and self._gpu
            if self._pretrained_model is None:
                model_type = self._model_type
            else:
                model_type = None

            self._model = models.CellposeModel(
                gpu=gpu,
                model_type=model_type,
                pretrained_model=self._pretrained_model
            )
            self._model.mkldnn = False
            self._model.net.mkldnn = False

            self._model_dropout = models.CellposeModel(
                gpu=gpu,
                model_type=model_type,
                pretrained_model=self._pretrained_model
            )
            self._model_dropout.mkldnn = False
            self._model_dropout.net.mkldnn = False

            self._model_dropout.net.load_model(
                self._model_dropout.pretrained_model,
                device=self._model_dropout.device
            )
            add_dropout(self._model_dropout.net)
            self._model_dropout.net.eval()
            self._transform = CellposeTransform(self._channels,
                                                self._channel_axis)

            self.refresh_model = False

        def _run_pred(self, img, *args, **kwargs):
            if self.refresh_model:
                self._model_init()

            x = self._transform(img)

            with torch.no_grad():
                y, _ = core.run_net(self._model_dropout.net, x)
                logits = torch.from_numpy(y[:, :, 2])
                probs = logits.sigmoid().numpy()

            return probs

        def _run_eval(self, img, *args, **kwargs):
            if self.refresh_model:
                self._model_init()

            seg, _, _ = self._model.eval(img, diameter=None,
                                         flow_threshold=None,
                                         channels=self._channels)
            return seg

    class CellposeTunable(CellposeSegmentation, FineTuningMethod):
        def __init__(self):
            super().__init__()

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

        def _get_transform(self):
            return self._transform

        def _fine_tune(self, train_data, train_labels, test_data, test_labels):
            self._model_init()

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

            self.refresh_model = True

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


class SimpleSegmentation(SegmentationMethod):
    def __init__(self):
        super().__init__()
        self._channel_axis = 2
        self._transform = None
        self._threshold = 0.5

    def _model_init(self):
        self._transform = BaseTransform(channel_axis=self._channel_axis)

    def _run_pred(self, img, *args, **kwargs):
        if self._transform is None:
            self._model_init()

        img_g = self._transform(img)
        img_g = img_g + 1e-3 * np.random.randn(*img_g.shape)
        img_g = (img_g - img_g.min()) / (img_g.max() - img_g.min() + 1e-12)
        return img_g

    def _run_eval(self, img, *args, **kwargs):
        if self._transform is None:
            self._model_init()

        img_g = self._transform(img)

        labels = skimage.measure.label(img_g > self._threshold)
        return labels


class SimpleTunable(SimpleSegmentation, FineTuningMethod):
    def __init__(self):
        super().__init__()

    def _get_transform(self):
        if self._transform is None:
            self._model_init()

        return self._transform

    def _fine_tune(self, train_data, train_labels, test_data, test_labels):
        if self._transform is None:
            self._model_init()
