from qtpy.QtWidgets import QWidget
from typing import Annotated, Literal
from pathlib import Path
from magicgui import magicgui

import napari_activelearning as al

from .microsam_activelearning import TunableMicroSAM


class TunableMicroSAMWidget(TunableMicroSAM, al.TunableWidget):
    def __init__(self):
        super().__init__()

    def _segmentation_parameters_widget(self):
        @magicgui(auto_call=True)
        def microsam_segmentation_parameters(
          checkpoint_path: Annotated[Path, {"widget_type": "FileEdit",
                                            "visible": False,
                                            "mode": "r"}] = Path(""),
          model_type: Literal["vit_l",
                              "vit_h",
                              "vit_b",
                              "vit_t",
                              "vit_l_lm",
                              "vit_b_lm",
                              "vit_t_lm",
                              "vit_l_em_organelles",
                              "vit_b_em_organelles",
                              "vit_t_em_organelles",
                              "vit_b_histopathology",
                              "vit_l_histopathology",
                              "vit_h_histopathology",
                              "vit_b_medical_imaging"] = "vit_b",
          gpu: bool = True):
            return dict(
                checkpoint_path=checkpoint_path,
                model_type=model_type,
                gpu=gpu
            )

        segmentation_parameter_names = [
                "checkpoint_path",
                "model_type",
                "gpu"
            ]

        return microsam_segmentation_parameters, segmentation_parameter_names

    def _finetuning_parameters_widget(self):
        @magicgui(auto_call=True)
        def microsam_finetuning_parameters(
          save_path: Annotated[Path, {"widget_type": "FileEdit",
                                      "mode": "d"}] = Path(""),
          model_name: str = "",
          batch_size: Annotated[int, {"widget_type": "SpinBox",
                                      "min": 1,
                                      "max": 1024}] = 8,
          learning_rate: Annotated[float, {"widget_type": "FloatSpinBox",
                                           "min": 1e-5,
                                           "max": 1.0,
                                           "step": 1e-5}] = 0.005,
          n_epochs: Annotated[int, {"widget_type": "SpinBox",
                                    "min": 1,
                                    "max": 10000}] = 20):
            return dict(
                batch_size=batch_size,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                save_path=save_path,
                model_name=model_name
            )

        finetuning_parameter_names = [
            "batch_size",
            "learning_rate",
            "n_epochs",
            "save_path",
            "model_name"
        ]

        return microsam_finetuning_parameters, finetuning_parameter_names

    def _check_parameter(self, parameter_val, parameter_key=None):
        if (((parameter_key in ("_checkpoint_path"))
             and not parameter_val.exists())
            or (isinstance(parameter_val, (int, float))
                and parameter_val < 0)):
            parameter_val = None

        return parameter_val

    def _fine_tune(self, train_dataloader, val_dataloader):
        super()._fine_tune(train_dataloader, val_dataloader)


def register_microsam():
    al.register_model("micro-sam", TunableMicroSAMWidget)
