import numpy as np
import torch
import time

from micro_sam import util
from micro_sam import automatic_segmentation as msas
import napari_activelearning as al


class TunableMicroSAM(al.TunableMethodWidget):
    def __init__(self):
        super(TunableMicroSAM, self).__init__()
        self._predictor = None
        self._amg = None

    def _model_init(self):
        if self._amg is not None:
            return

        (self._sam_predictor,
         self._sam_instance_segmenter) = msas.get_predictor_and_segmenter(
            model_type='vit_t',
            device=util.get_device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu"),
            amg=True,
            checkpoint=None,
            stability_score_offset=1.0
        )

        (self._sam_predictor_dropout,
         self._sam_instance_segmenter_dropout) =\
            msas.get_predictor_and_segmenter(
                model_type='vit_t',
                device=util.get_device("cuda"
                                       if torch.cuda.is_available()
                                       else "cpu"),
                amg=True,
                checkpoint=None,
                stability_score_offset=1.0)

        al.add_dropout(self._sam_predictor_dropout.model)


    def _get_transform(self):
        return lambda x: x

    def _run_pred(self, img, *args, **kwargs):
        self._model_init()

        e_time = time.perf_counter()
        img_embeddings = util.precompute_image_embeddings(
            predictor=self._sam_predictor_dropout,
            input_=img,
            save_path=None,
            ndim=2,
            tile_shape=None,
            halo=None,
            verbose=False,
        )
        e_time = time.perf_counter() - e_time

        e_time = time.perf_counter()
        self._sam_instance_segmenter_dropout.initialize(
            image=img,
            image_embeddings=img_embeddings
        )
        e_time = time.perf_counter() - e_time

        e_time = time.perf_counter()
        masks = self._sam_instance_segmenter_dropout.generate()
        e_time = time.perf_counter() - e_time

        e_time = time.perf_counter()
        probs = np.zeros(img.shape[:2], dtype=np.float32)
        for mask in masks:
            probs = np.where(
                mask["segmentation"],
                mask["predicted_iou"],
                probs
            )
        e_time = time.perf_counter() - e_time

        probs = torch.from_numpy(probs).sigmoid().numpy()

        return probs

    def _run_eval(self, img, *args, **kwargs):
        self._model_init()

        e_time = time.perf_counter()
        segmentation_mask = msas.automatic_instance_segmentation(
            predictor=self._sam_predictor,
            segmenter=self._sam_instance_segmenter,
            input_path=img,
            ndim=2,
            verbose=False
        )
        e_time = time.perf_counter() - e_time

        return segmentation_mask

    def _fine_tune(self, train_data, train_labels, test_data, test_labels):
        self._model_init()
        return None


def register_microsam():
    al.register_model("micro-sam", TunableMicroSAM)
