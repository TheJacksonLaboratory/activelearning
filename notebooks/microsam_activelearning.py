import os
import numpy as np
import torch
import time

from torchvision import transforms
from torch_em.transform.label import PerObjectDistanceTransform

from micro_sam import util
from micro_sam import automatic_segmentation as msas
import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model
import napari_activelearning as al


class TunableMicroSAM(al.TunableMethodWidget):
    def __init__(self):
        super(TunableMicroSAM, self).__init__()
        self._sam_predictor = None
        self._sam_instance_segmenter = None
        self.checkpoint_path = None
        self.model_type = "vit_t"
        self.lr = 1e-5
        self.n_objects_per_batch = 5
        self.n_epochs = 5
        self.checkpoint_name = f"{self.model_type}/finetuned"
        self.save_root = "./finetuned_models/checkpoints"
        self.export_path = "./finetuned_models"

    def _model_init(self):
        if self._sam_predictor is not None:
            return

        (self._sam_predictor,
         self._sam_instance_segmenter) = msas.get_predictor_and_segmenter(
            model_type=self.model_type,
            device=util.get_device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu"),
            amg=True,
            checkpoint=self.checkpoint_path,
            stability_score_offset=1.0
        )

        (self._sam_predictor_dropout,
         self._sam_instance_segmenter_dropout) =\
            msas.get_predictor_and_segmenter(
                model_type=self.model_type,
                device=util.get_device("cuda"
                                       if torch.cuda.is_available()
                                       else "cpu"),
                amg=True,
                checkpoint=self.checkpoint_path,
                stability_score_offset=1.0)

        al.add_dropout(self._sam_predictor_dropout.model)

    def _get_transform(self):
        # Ensure labels are squeezed when these are not actual 3D arrays.
        def squeeze_labels(x):
            return x.squeeze()

        def to_uint8(x):
            x_float = x.astype(np.float32)
            x_rescaled = (x_float - x_float.min())\
                / (x_float.max() - x_float.min())
            return (255.0 * x_rescaled).astype(np.uint8)

        input_transform = transforms.Compose([
            to_uint8,
        ])

        label_transform = transforms.Compose([
            transforms.Lambda(squeeze_labels),
            PerObjectDistanceTransform(
                distances=True,
                boundary_distances=True,
                directed_distances=False,
                foreground=True,
                instances=True,
                min_size=25
            )
        ])

        return input_transform, label_transform

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

    def _fine_tune(self, train_dataloader, val_dataloader) -> bool:
        self._model_init()

        train_dataloader.shuffle = True
        val_dataloader.shuffle = False

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # training settings:
        freeze_parts = None

        # all the stuff we need for training
        scheduler_kwargs = {
            "mode": "min",
            "factor": 0.9,
            "patience": 10,
            "verbose": True
        }

        # Run training.
        sam_training.train_sam(
            name=self.checkpoint_name,
            model_type=self.model_type,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            with_segmentation_decoder=True,
            early_stopping=10,
            n_objects_per_batch=self.n_objects_per_batch,
            checkpoint_path=self.checkpoint_path,
            freeze=freeze_parts,
            device=device,
            lr=self.lr,
            n_epochs=self.n_epochs,
            save_root=self.save_root,
            scheduler_kwargs=scheduler_kwargs,
            verify_n_labels_in_loader=2,
            save_every_kth_epoch=10,
            peft_kwargs=None,
        )

        if self.export_path is not None:
            self.checkpoint_path = os.path.join(
                "" if self.save_root is None else self.save_root,
                "checkpoints",
                self.checkpoint_name,
                "best.pt"
            )
            export_custom_sam_model(
                checkpoint_path=self.checkpoint_path,
                model_type=self.model_type,
                save_path=self.export_path,
            )

            self._sam_predictor = None
            self._sam_instance_segmenter = None

        return True


def register_microsam():
    al.register_model("micro-sam", TunableMicroSAM)
