from typing import Annotated, Literal
from pathlib import Path
from magicgui import magicgui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QGridLayout, QScrollArea, QCheckBox

from functools import partial
from ._models import USING_CELLPOSE, SimpleTunable

if USING_CELLPOSE:
    from ._models import CellposeTunable

    def cellpose_segmentation_parameters_widget():
        @magicgui(auto_call=True)
        def cellpose_segmentation_parameters(
          channel_axis: Annotated[int, {"widget_type": "SpinBox",
                                        "min": 0,
                                        "max": 2**16}] = 2,
          channels: tuple[int, int] = (0, 0),
          pretrained_model: Annotated[Path, {"widget_type": "FileEdit",
                                             "visible": False,
                                             "mode": "r"}] = Path(""),
          model_type: Literal["custom",
                              "cyto",
                              "cyto2",
                              "cyto3",
                              "nuclei",
                              "tissuenet_cp3"] = "cyto3",
          gpu: bool = True
          ):
            return dict(
                channel_axis=channel_axis,
                channels=channels,
                pretrained_model=pretrained_model,
                model_type=model_type,
                gpu=gpu
            )

        segmentation_parameter_names = [
                "channel_axis",
                "channels",
                "pretrained_model",
                "model_type",
                "gpu"
            ]

        return cellpose_segmentation_parameters, segmentation_parameter_names

    def cellpose_finetuning_parameters_widget():
        @magicgui(auto_call=True)
        def cellpose_finetuning_parameters(
          weight_decay: Annotated[float, {"widget_type": "FloatSpinBox",
                                          "min": 0.0,
                                          "max": 1.0,
                                          "step": 1e-5}] = 1e-5,
          momentum: Annotated[float, {"widget_type": "FloatSpinBox",
                                      "min": 0,
                                      "max": 1,
                                      "step": 1e-2}] = 0.9,
          SGD: bool = False,
          rgb: bool = False,
          normalize: bool = True,
          compute_flows: bool = False,
          save_path: Annotated[Path, {"widget_type": "FileEdit",
                                      "mode": "d"}] = Path(""),
          save_every: Annotated[int, {"widget_type": "SpinBox",
                                      "min": 1,
                                      "max": 10000}] = 100,
          nimg_per_epoch: Annotated[int, {"widget_type": "SpinBox",
                                          "min": -1,
                                          "max": 2**16}] = -1,
          nimg_test_per_epoch: Annotated[int, {"widget_type": "SpinBox",
                                               "min": -1,
                                               "max": 2**16}] = -1,
          rescale: bool = True,
          scale_range: Annotated[int, {"widget_type": "SpinBox",
                                       "min": -1,
                                       "max": 2**16}] = -1,
          bsize: Annotated[int, {"widget_type": "SpinBox",
                                 "min": 64,
                                 "max": 2**16}] = 224,
          min_train_masks: Annotated[int, {"widget_type": "SpinBox",
                                           "min": 1,
                                           "max": 2**16}] = 5,
          model_name: str = "",
          batch_size: Annotated[int, {"widget_type": "SpinBox",
                                      "min": 1,
                                      "max": 1024}] = 8,
          learning_rate: Annotated[float, {"widget_type": "FloatSpinBox",
                                           "min": 1e-10,
                                           "max": 1.0,
                                           "step": 1e-3}] = 0.005,
          n_epochs: Annotated[int, {"widget_type": "SpinBox",
                                    "min": 1,
                                    "max": 10000}] = 20):
            return dict(
                batch_size=batch_size,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                weight_decay=weight_decay,
                momentum=momentum,
                SGD=SGD,
                rgb=rgb,
                normalize=normalize,
                compute_flows=compute_flows,
                save_path=save_path,
                save_every=save_every,
                nimg_per_epoch=nimg_per_epoch,
                nimg_test_per_epoch=nimg_test_per_epoch,
                rescale=rescale,
                scale_range=scale_range,
                bsize=bsize,
                min_train_masks=min_train_masks,
                model_name=model_name
            )

        finetuning_parameter_names = [
            "batch_size",
            "learning_rate",
            "n_epochs",
            "weight_decay",
            "momentum",
            "SGD",
            "rgb",
            "normalize",
            "compute_flows",
            "save_path",
            "save_every",
            "nimg_per_epoch",
            "nimg_test_per_epoch",
            "rescale",
            "scale_range",
            "bsize",
            "min_train_masks",
            "model_name"
        ]

        return cellpose_finetuning_parameters, finetuning_parameter_names

    class CellposeTunableWidget(CellposeTunable, QWidget):
        def __init__(self):
            super().__init__()

            (self._segmentation_parameters,
             segmentation_parameter_names) =\
                cellpose_segmentation_parameters_widget()

            (self._finetuning_parameters,
             finetuning_parameter_names) =\
                cellpose_finetuning_parameters_widget()

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
                self._segmentation_parameters.__getattr__(par_name)\
                                             .changed.connect(
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
            if (((parameter_key in {"_save_path", "_pretrained_model"})
                 and not parameter_val.exists())
               or (isinstance(parameter_val, (int, float))
                   and parameter_val < 0)):
                parameter_val = None

            if parameter_key == "_model_type":
                if parameter_val == "custom":
                    self._segmentation_parameters\
                        .pretrained_model\
                        .visible = True
                else:
                    self._segmentation_parameters\
                        .pretrained_model\
                        .visible = False
                    self._pretrained_model = None

            if getattr(self, parameter_key) != parameter_val:
                self.refresh_model = True
                setattr(self, parameter_key, parameter_val)

        def _show_segmentation_parameters(self, show: bool):
            self._segmentation_parameters_scr.setVisible(show)

        def _show_finetuning_parameters(self, show: bool):
            self._finetuning_parameters_scr.setVisible(show)

        def _fine_tune(self, train_data, train_labels, test_data, test_labels):
            super()._fine_tune(train_data, train_labels, test_data,
                               test_labels)
            self._segmentation_parameters.pretrained_model.value =\
                self._pretrained_model


def simple_segmentation_parameters_widget():
    @magicgui(auto_call=True)
    def simple_segmentation_parameters(
        channel_axis: Annotated[int, {"widget_type": "SpinBox", "min": 0,
                                      "max": 2**16}] = 2,
        threshold: Annotated[float, {"widget_type": "FloatSpinBox",
                                     "min": 0.0,
                                     "max": 1.0,
                                     "step": 1e-5}] = 0.5,
    ):
        return dict(channel_axis=channel_axis, threshold=threshold)

    segmentation_parameter_names = [
            "channel_axis",
            "threshold"
        ]

    return simple_segmentation_parameters, segmentation_parameter_names


class SimpleTunableWidget(SimpleTunable, QWidget):
    def __init__(self):
        super().__init__()

        (self._segmentation_parameters,
            segmentation_parameter_names) =\
            simple_segmentation_parameters_widget()

        self.advanced_segmentation_options_chk = QCheckBox(
            "Advanced segmentation parameters"
        )

        self.advanced_segmentation_options_chk.setChecked(False)
        self.advanced_segmentation_options_chk.toggled.connect(
            self._show_segmentation_parameters
        )

        self._segmentation_parameters_scr = QScrollArea()
        self._segmentation_parameters_scr.setWidgetResizable(True)
        self._segmentation_parameters_scr.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )
        self._segmentation_parameters_scr.setWidget(
            self._segmentation_parameters.native
        )

        for par_name in segmentation_parameter_names:
            self._segmentation_parameters.__getattr__(par_name).changed\
                                                               .connect(
                partial(self._set_parameter, parameter_key="_" + par_name)
            )

        self.parameters_lyt = QGridLayout()
        self.parameters_lyt.addWidget(
            self.advanced_segmentation_options_chk, 0, 0
        )
        self.parameters_lyt.addWidget(
            self._segmentation_parameters_scr, 1, 0, 1, 2
        )
        self.setLayout(self.parameters_lyt)

        self._segmentation_parameters_scr.hide()

    def _set_parameter(self, parameter_val, parameter_key=None):
        setattr(self, parameter_key, parameter_val)

    def _show_segmentation_parameters(self, show: bool):
        self._segmentation_parameters_scr.setVisible(show)
