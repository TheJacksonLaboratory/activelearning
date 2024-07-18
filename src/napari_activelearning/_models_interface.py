from typing import Annotated, Literal
from pathlib import Path
from magicgui import magicgui
from qtpy.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QCheckBox

from functools import partial

from ._models import USING_CELLPOSE

if USING_CELLPOSE:
    from ._models import CellposeTunable

    def cellpose_segmentation_parameters_widget():
        @magicgui(auto_call=True)
        def cellpose_segmentation_parameters(
          channel_axis: Annotated[int, {"widget_type": "SpinBox",
                                        "min": 0,
                                        "max": 2**16}] = 2,
          channels: tuple[int, int] = (0, 0),
          pretrained_model: Path = Path(""),
          model_type: Literal["cyto", "nuclei", "tissuenet_cp3"] = "cyto",
          gpu: bool = True
          ):
            return dict(
                channel_axis=channel_axis,
                channels=channels,
                pretrained_model=pretrained_model,
                model_type=model_type,
                gpu=gpu
            )

        return cellpose_segmentation_parameters

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
                                    "max": 10000}] = 2000):
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
        return cellpose_finetuning_parameters

    class CellposeTunableWidget(QWidget, CellposeTunable):
        def __init__(self):
            super().__init__()

            segmentation_parameter_names = [
                "channel_axis",
                "channels",
                "pretrained_model",
                "model_type",
                "gpu"
            ]
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

            cellpose_segmentation_parameters =\
                cellpose_segmentation_parameters_widget()

            self._segmentation_parameters_widget =\
                cellpose_segmentation_parameters.native

            cellpose_finetuning_parameters =\
                cellpose_finetuning_parameters_widget()

            self._finetuning_parameters_widget =\
                cellpose_finetuning_parameters.native

            for par_name in segmentation_parameter_names:
                cellpose_segmentation_parameters.__getattr__(par_name).changed\
                                                                      .connect(
                    partial(self._set_parameter, parameter_key="_" + par_name)
                )

            for par_name in finetuning_parameter_names:
                cellpose_finetuning_parameters.__getattr__(par_name).changed\
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
                self._segmentation_parameters_widget, 1, 0, 1, 2
            )
            self.parameters_lyt.addWidget(
                self._finetuning_parameters_widget, 3, 0, 1, 2
            )
            self.setLayout(self.parameters_lyt)

            self._segmentation_parameters_widget.hide()
            self._finetuning_parameters_widget.hide()

        def _set_parameter(self, parameter_val, parameter_key=None):
            if ((isinstance(parameter_val, (str, Path)) and not parameter_val)
               or (isinstance(parameter_val, (int, float))
                   and parameter_val < 0)):
                parameter_val = None

            self.__setattr__(parameter_key, parameter_val)

        def _show_segmentation_parameters(self, show: bool):
            if self._segmentation_parameters_widget is not None:
                self._segmentation_parameters_widget.setVisible(show)

        def _show_finetuning_parameters(self, show: bool):
            if self._finetuning_parameters_widget is not None:
                self._finetuning_parameters_widget.setVisible(show)
