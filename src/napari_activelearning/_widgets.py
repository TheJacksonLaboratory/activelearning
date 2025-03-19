from ._models import TunableMethod
from ._interface import (ImageGroupsManagerWidget,
                         LabelsManagerWidget,
                         AcquisitionFunctionWidget)

from ._models_impl import USING_CELLPOSE
from ._models_impl_interface import SimpleTunableWidget


CURRENT_IMAGE_GROUPS_MANAGER = None
CURRENT_LABEL_GROUPS_MANAGER = None
CURRENT_SEGMENTATION_METHOD = None
CURRENT_ACQUISITION_FUNCTION = None

models_registry: dict[str] = {
    "None selected": None
}


def register_model(model_name: str, model: TunableMethod):
    global CURRENT_ACQUISITION_FUNCTION
    if model_name in models_registry:
        return

    models_registry[model_name] = model

    if CURRENT_ACQUISITION_FUNCTION is not None:
        CURRENT_ACQUISITION_FUNCTION = AcquisitionFunctionWidget(
            image_groups_manager=get_image_groups_manager_widget(),
            labels_manager=get_label_groups_manager_widget(),
            tunable_segmentation_methods=models_registry,
        )


register_model("simple", SimpleTunableWidget)
if USING_CELLPOSE:
    from ._models_impl_interface import CellposeTunableWidget
    register_model("cellpose", CellposeTunableWidget)


def get_image_groups_manager_widget():
    global CURRENT_IMAGE_GROUPS_MANAGER

    if CURRENT_IMAGE_GROUPS_MANAGER is None:
        CURRENT_IMAGE_GROUPS_MANAGER = ImageGroupsManagerWidget(
            default_axis_labels="TCZYX"
        )

    return CURRENT_IMAGE_GROUPS_MANAGER


def get_label_groups_manager_widget():
    global CURRENT_LABEL_GROUPS_MANAGER

    if CURRENT_LABEL_GROUPS_MANAGER is None:
        CURRENT_LABEL_GROUPS_MANAGER = LabelsManagerWidget()

    return CURRENT_LABEL_GROUPS_MANAGER


def get_acquisition_function_widget():
    global CURRENT_ACQUISITION_FUNCTION

    if CURRENT_ACQUISITION_FUNCTION is None:
        CURRENT_ACQUISITION_FUNCTION = AcquisitionFunctionWidget(
            image_groups_manager=get_image_groups_manager_widget(),
            labels_manager=get_label_groups_manager_widget(),
            tunable_segmentation_methods=models_registry,
        )

    return CURRENT_ACQUISITION_FUNCTION


def get_active_learning_widget():
    return [
        get_image_groups_manager_widget(),
        get_acquisition_function_widget(),
        get_label_groups_manager_widget()
    ]
