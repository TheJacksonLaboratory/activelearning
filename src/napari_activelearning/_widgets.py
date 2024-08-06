from ._interface import (ImageGroupsManagerWidget,
                         LabelsManagerWidget,
                         AcquisitionFunctionWidget)

from ._models import USING_CELLPOSE
from ._models_interface import SimpleTunableWidget

if USING_CELLPOSE:
    from ._models_interface import CellposeTunableWidget

    SEGMENTATION_METHOD_CLASS = CellposeTunableWidget

else:
    SEGMENTATION_METHOD_CLASS = SimpleTunableWidget

CURRENT_IMAGE_GROUPS_MANAGER = None
CURRENT_LABEL_GROUPS_MANAGER = None
CURRENT_SEGMENTATION_METHOD = None
CURRENT_ACQUISITION_FUNCTION = None


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
        segmentation_method = SEGMENTATION_METHOD_CLASS()

        CURRENT_ACQUISITION_FUNCTION = AcquisitionFunctionWidget(
            image_groups_manager=get_image_groups_manager_widget(),
            labels_manager=get_label_groups_manager_widget(),
            tunable_segmentation_method=segmentation_method,
        )

    return CURRENT_ACQUISITION_FUNCTION


def get_active_learning_widget():
    return [
        get_image_groups_manager_widget(),
        get_acquisition_function_widget(),
        get_label_groups_manager_widget()
    ]
