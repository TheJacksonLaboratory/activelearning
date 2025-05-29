from typing import Annotated, Literal
from pathlib import Path
from magicgui import magicgui

from ._models_impl import USING_CELLPOSE, SimpleTunable
from ._models import TunableWidget

if USING_CELLPOSE:
    from ._models_impl import CellposeTunable

    class CellposeTunableWidget(CellposeTunable, TunableWidget):
        def __init__(self):
            super().__init__()

        @staticmethod
        def _segmentation_parameters_widget():
            @magicgui(auto_call=True)
            def cellpose_segmentation_parameters(
              channels: tuple[int, int] = (0, 0),
              pretrained_model: Annotated[Path, {"widget_type": "FileEdit",
                                                 "visible": False,
                                                 "mode": "r"}] = Path(""),
              model_type: Literal["custom",
                                  "cyto",
                                  "cyto2",
                                  "cyto3",
                                  "nuclei",
                                  "tissuenet_cp3",
                                  "livecell_cp3",
                                  "yeast_PhC_cp3",
                                  "yeast_BF_cp3",
                                  "bact_phase_cp3",
                                  "bact_fluor_cp3",
                                  "deepbacs_cp3",
                                  "cyto2_cp3",
                                  "CP",
                                  "CPx",
                                  "TN1",
                                  "TN2",
                                  "TN3",
                                  "LC1",
                                  "LC2",
                                  "LC3",
                                  "LC"] = "cyto3",
              gpu: bool = True):
                return dict(
                    channels=channels,
                    pretrained_model=pretrained_model,
                    model_type=model_type,
                    gpu=gpu
                )

            segmentation_parameter_names = [
                    "channels",
                    "pretrained_model",
                    "model_type",
                    "gpu"
                ]

            return (cellpose_segmentation_parameters,
                    segmentation_parameter_names)

        @staticmethod
        def _finetuning_parameters_widget():
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
                                               "step": 1e-10}] = 0.005,
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

        def _check_parameter(self, parameter_val, parameter_key=None):
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

            return parameter_val

        def _fine_tune(self, train_dataloader, val_dataloader):
            super()._fine_tune(train_dataloader, val_dataloader)
            self._segmentation_parameters.model_type.value = "custom"
            self._segmentation_parameters.pretrained_model.value =\
                self._pretrained_model


class SimpleTunableWidget(SimpleTunable, TunableWidget):
    def __init__(self):
        super().__init__()

    def _segmentation_parameters_widget(self):
        @magicgui(auto_call=True)
        def simple_segmentation_parameters(
            threshold: Annotated[float, {"widget_type": "FloatSpinBox",
                                         "min": 0.0,
                                         "max": 1.0,
                                         "step": 1e-5}] = 0.5,
        ):
            return dict(threshold=threshold)

        segmentation_parameter_names = [
                "threshold"
            ]

        return simple_segmentation_parameters, segmentation_parameter_names

    def _finetuning_parameters_widget(self):
        return None, None
