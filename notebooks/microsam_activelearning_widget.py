from functools import partial

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QCheckBox, QScrollArea, QGridLayout
from typing import Annotated, Literal
from pathlib import Path
from magicgui import magicgui

import napari_activelearning as al

from microsam_activelearning import TunableMicroSAM


def microsam_segmentation_parameters_widget():
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


def microsam_finetuning_parameters_widget():
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


class TunableMicroSAMWidget(TunableMicroSAM, QWidget):
    def __init__(self):
        super().__init__()

        (self._segmentation_parameters,
            segmentation_parameter_names) =\
            microsam_segmentation_parameters_widget()

        (self._finetuning_parameters,
            finetuning_parameter_names) =\
            microsam_finetuning_parameters_widget()

        self.advanced_segmentation_options_chk = QCheckBox(
            "Advanced segmentation parameters"
        )

        self.advanced_segmentation_options_chk.setChecked(False)
        self.advanced_segmentation_options_chk.toggled.connect(
            self._show_segmentation_parameters
        )

        self.advanced_finetuning_options_chk = QCheckBox(
            "Advanced fine tuning parameters"
        )

        self.advanced_finetuning_options_chk.setChecked(False)
        self.advanced_finetuning_options_chk.toggled.connect(
            self._show_finetuning_parameters
        )

        self._segmentation_parameters_scr = QScrollArea()
        self._segmentation_parameters_scr.setWidgetResizable(True)
        self._segmentation_parameters_scr.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        self._segmentation_parameters_scr.setWidget(
            self._segmentation_parameters.native
        )

        self._finetuning_parameters_scr = QScrollArea()
        self._finetuning_parameters_scr.setWidgetResizable(True)
        self._finetuning_parameters_scr.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        self._finetuning_parameters_scr.setWidget(
            self._finetuning_parameters.native
        )

        for par_name in segmentation_parameter_names:
            self._segmentation_parameters.__getattr__(par_name).changed\
                                                               .connect(
                partial(self._set_parameter, parameter_key="_" + par_name)
            )

        for par_name in finetuning_parameter_names:
            self._finetuning_parameters.__getattr__(par_name).changed\
                                                             .connect(
                partial(self._set_parameter, parameter_key="_" + par_name)
            )

        self.parameters_lyt = QGridLayout()
        self.parameters_lyt.addWidget(
            self.advanced_segmentation_options_chk, 0, 0
        )
        self.parameters_lyt.addWidget(
            self.advanced_finetuning_options_chk, 2, 0
        )
        self.parameters_lyt.addWidget(
            self._segmentation_parameters_scr, 1, 0, 1, 2
        )
        self.parameters_lyt.addWidget(
            self._finetuning_parameters_scr, 3, 0, 1, 2
        )
        self.setLayout(self.parameters_lyt)

        self._segmentation_parameters_scr.hide()
        self._finetuning_parameters_scr.hide()

    def _set_parameter(self, parameter_val, parameter_key=None):
        if (((parameter_key in ("_checkpoint_path"))
             and not parameter_val.exists())
            or (isinstance(parameter_val, (int, float))
                and parameter_val < 0)):
            parameter_val = None

        if getattr(self, parameter_key) != parameter_val:
            self.refresh_model = True
            setattr(self, parameter_key, parameter_val)

    def _show_segmentation_parameters(self, show: bool):
        self._segmentation_parameters_scr.setVisible(show)

    def _show_finetuning_parameters(self, show: bool):
        self._finetuning_parameters_scr.setVisible(show)

    def _fine_tune(self, train_dataloader, val_dataloader):
        super()._fine_tune(train_dataloader, val_dataloader)


def register_microsam():
    al.register_model("micro-sam", TunableMicroSAMWidget)
