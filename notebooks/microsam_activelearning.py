import os
import numpy as np
import torch

from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.transform.augmentation import get_augmentations
from torch_em.util.util import ensure_tensor_with_channels

from micro_sam import util
from micro_sam import automatic_segmentation as msas
from micro_sam import instance_segmentation as msis
import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model

import napari_activelearning as al


class AugmentEnsureInputs:
    def __init__(self):
        self.label_transform = PerObjectDistanceTransform(
            distances=True,
            boundary_distances=True,
            directed_distances=False,
            foreground=True,
            instances=True,
            min_size=25
        )

        self.augmentations = get_augmentations(ndim=2)

    def __call__(self, inputs, labels):
        inputs_tr = inputs.copy()
        labels_tr = labels.copy()

        if inputs_tr.ndim == 3 and inputs_tr.shape[-1] == 1:
            inputs_tr = inputs_tr.squeeze()

        labels_tr = self.label_transform(labels_tr)

        inputs_tr, labels_tr = self.augmentations(inputs_tr, labels_tr)

        inputs_tr = ensure_tensor_with_channels(inputs_tr, ndim=2,
                                                dtype=torch.float32)
        labels_tr = ensure_tensor_with_channels(labels_tr, ndim=2,
                                                dtype=torch.float32)

        if (inputs_tr.ndim == 3
           and (inputs_tr.shape[-1] == 1
                or inputs_tr.shape[-1] == 3)):
            if isinstance(inputs_tr, np.ndarray):
                inputs_tr = np.moveaxis(inputs_tr, -1, 0)
            elif isinstance(inputs_tr, torch.Tensor):
                inputs_tr = torch.transpose(inputs_tr, -1, 0)

        inputs_tr = inputs_tr.contiguous()
        labels_tr = labels_tr.contiguous()
        return (inputs_tr, labels_tr)


class TunableMicroSAM(al.TunableMethod):
    model_axes = "YXC"

    def __init__(self):
        super(TunableMicroSAM, self).__init__()
        self._sam_predictor = None
        self._sam_instance_segmenter = None
        self._checkpoint_path = None
        self._model_type = "vit_b"
        self._n_epochs = 5
        self._learning_rate = 1e-5
        self._model_name = "vit_b_msam_ft"
        self._save_path = "./finetuned_models"
        self._gpu = True

        self._pred_iou_thresh = 0.88
        self._stability_score_thresh = 0.95
        self._box_nms_thresh = 0.7

        self.refresh_model = True

    def _model_init(self):
        if not self.refresh_model:
            return

        device = util.get_device(
            "cuda" if torch.cuda.is_available() and self._gpu else "cpu"
        )

        (self._sam_predictor,
         self._sam_instance_segmenter) = msas.get_predictor_and_segmenter(
            model_type=self._model_type,
            device=device,
            checkpoint=self._checkpoint_path
        )

        (self._sam_predictor_dropout,
         self._sam_instance_segmenter_dropout) =\
            msas.get_predictor_and_segmenter(
                model_type=self._model_type,
                device=device,
                checkpoint=self._checkpoint_path)

        al.add_dropout(self._sam_predictor_dropout.model.image_encoder)
        if isinstance(self._sam_instance_segmenter_dropout,
                      msas.InstanceSegmentationWithDecoder):
            al.add_dropout(self._sam_instance_segmenter_dropout._decoder)

        self.refresh_model = False

    def get_train_transform(self, *args, **kwargs):
        # Ensure labels are squeezed when these are not actual 3D arrays.
        mode_transforms = {
            ("images", "labels"): AugmentEnsureInputs()
        }
        return mode_transforms

    def get_inference_transform(self, *args, **kwargs):
        # Ensure labels are squeezed when these are not actual 3D arrays.
        mode_transforms = {
            ("images", ): lambda x: x
        }
        return mode_transforms

    def _run_pred(self, img, *args, **kwargs):
        self._model_init()

        img_embeddings = util.precompute_image_embeddings(
            predictor=self._sam_predictor_dropout,
            input_=img.squeeze(),
            save_path=None,
            ndim=2,
            tile_shape=None,
            halo=None,
            verbose=False,
        )

        if isinstance(self._sam_instance_segmenter_dropout,
                      msis.AutomaticMaskGenerator):
            self._sam_instance_segmenter_dropout.initialize(
                image=img.squeeze(),
                image_embeddings=img_embeddings
            )

            generate_kwargs = {}
            if not isinstance(self._sam_instance_segmenter_dropout,
                            msas.InstanceSegmentationWithDecoder):
                generate_kwargs["pred_iou_thresh"] = self._pred_iou_thresh
                generate_kwargs["stability_score_thresh"] =\
                    self._stability_score_thresh
                generate_kwargs["box_nms_thresh"] = self._box_nms_thresh

            masks = self._sam_instance_segmenter_dropout.generate(
                **generate_kwargs
            )

            probs = np.zeros(img.squeeze().shape[:2], dtype=np.float32)
            for mask in masks:
                probs = np.where(
                    mask["segmentation"],
                    mask["predicted_iou"],
                    probs
                )

            probs = torch.from_numpy(probs).sigmoid().numpy()
        else:
            self._sam_instance_segmenter_dropout.initialize(
                image=img.squeeze(),
                image_embeddings=img_embeddings
            )

            probs = self._sam_instance_segmenter_dropout._foreground.copy()

        return probs

    def _run_eval(self, img, *args, **kwargs):
        self._model_init()

        generate_kwargs = {}
        if not isinstance(self._sam_instance_segmenter_dropout,
                          msas.InstanceSegmentationWithDecoder):
            generate_kwargs["pred_iou_thresh"] = self._pred_iou_thresh
            generate_kwargs["stability_score_thresh"] =\
                self._stability_score_thresh
            generate_kwargs["box_nms_thresh"] = self._box_nms_thresh

        segmentation_mask = msas.automatic_instance_segmentation(
            predictor=self._sam_predictor,
            segmenter=self._sam_instance_segmenter,
            input_path=img.squeeze(),
            ndim=2,
            verbose=False,
            **generate_kwargs
        )

        return segmentation_mask

    def _fine_tune(self, train_dataloader, val_dataloader) -> bool:
        # self._model_init()

        train_dataloader.shuffle = True
        val_dataloader.shuffle = True

        device = "cuda" if torch.cuda.is_available() and self._gpu else "cpu"

        # Run training.
        sam_training.train_sam(
            name=self._model_name,
            save_root=self._save_path,
            model_type=self._model_type,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            n_epochs=self._n_epochs,
            lr=self._learning_rate,
            n_objects_per_batch=5,
            with_segmentation_decoder=True,
            device=device,
        )

        export_source_path = os.path.join(
            self._save_path,
            "checkpoints",
            self._model_name,
            "best.pt"
        )

        self._checkpoint_path = os.path.join(
            self._save_path,
            self._model_name + ".pth"
        )

        export_custom_sam_model(
            checkpoint_path=export_source_path,
            model_type=self._model_type,
            save_path=self._checkpoint_path,
            with_segmentation_decoder=True
        )

        self.refresh_model = True

        return True


class TunableMicroSAMWidget(TunableMicroSAM, al.TunableWidget):
    def __init__(self):
        super(TunableMicroSAMWidget, self).__init__()


def register_microsam():
    al.register_model("micro-sam", TunableMicroSAMWidget)
