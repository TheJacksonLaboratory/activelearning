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
            device=util.get_device("cpu"),
            amg=True,
            checkpoint=None,
            stability_score_offset=1.0
        )

    def _get_transform(self):
        return lambda x: x

    def _run_pred(self, img, *args, **kwargs):
        self._model_init()

        segmentation_mask = msas.automatic_instance_segmentation(
            predictor=self._sam_predictor,
            segmenter=self._sam_instance_segmenter,
            input_path=img,
            ndim=2,
            verbose=False
        )

        return segmentation_mask

    def _run_eval(self, img, *args, **kwargs):
        self._model_init()

        segmentation_mask = msas.automatic_instance_segmentation(
            predictor=self._sam_predictor,
            segmenter=self._sam_instance_segmenter,
            input_path=img,
            ndim=2,
            verbose=False
        )

        return segmentation_mask

    def _fine_tune(self, train_data, train_labels, test_data, test_labels):
        self._model_init()
        return None


def register_microsam():
    al.register_model("micro-sam", TunableMicroSAM)
