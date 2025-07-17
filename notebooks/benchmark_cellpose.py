import argparse
import napari
import napari_activelearning as al

from cellpose.metrics import average_precision
from sklearn.metrics import f1_score


def compute_metrics(ground_truth, pred, thresholds=None):
    if thresholds is None:
        thresholds = [0.5, 0.75, 0.9]

    ap, tp, fp, fn = average_precision(ground_truth, pred, thresholds)

    f1 = f1_score(ground_truth.flatten(), pred.flatten())

    return dict(f1=f1, avg_prec=ap, tp=tp, fp=fp, fn=fn)


def run_fine_tuning(input_filenames, ground_truth_filenames, viewer,
                    al_widget):
    image_groups_mgr = al_widget.widget(0)
    acquisition_fun_mgr = al_widget.widget(1)
    min_size = {"Y": float("inf"), "X": float("inf")}

    in_layer_list = []
    gt_layer_list = []

    for in_fn, gt_fn in zip(input_filenames, ground_truth_filenames):
        in_layer_list.append(
            viewer.open(in_fn, plugin="napari", layer_type="image")[0]
        )
        gt_layer_list.append(
            viewer.open(gt_fn, plugin="napari", layer_type="labels")[0]
        )

        viewer.layers.selection.clear()
        viewer.layers.selection.add(in_layer_list[-1])
        viewer.layers.selection.add(gt_layer_list[-1])

        image_groups_mgr.create_group()

        image_groups_mgr._active_image_group.child(0).setSelected(True)
        image_groups_mgr.image_groups_editor.edit_axes_le.setText("YXC")
        image_groups_mgr.image_groups_editor.update_source_axes()

        image_groups_mgr._active_image_group.child(1).setSelected(True)
        image_groups_mgr.image_groups_editor.use_as_labels_chk.setChecked(True)

        min_size["Y"] = min(in_layer_list.data.shape[0], min_size["Y"])
        min_size["X"] = min(in_layer_list.data.shape[1], min_size["X"])

    acquisition_fun_mgr.patch_sizes_mspn.sizes = {"Y": min_size["Y"],
                                                  "X": min_size["X"]}
    acquisition_fun_mgr.max_samples_spn.setValue(2)

    acquisition_fun_mgr.methods_cmb.setCurrentIndex(2)
    acquisition_fun_mgr.tunable_segmentation_method\
                       ._segmentation_parameters\
                       .channels.value = (0, 1)
    acquisition_fun_mgr.tunable_segmentation_method\
                       ._segmentation_parameters\
                       .model_type.value = "cyto3"

    _ = acquisition_fun_mgr.compute_acquisition_layers(run_all=True)

    img_groups_list = map(
        lambda c_idx: image_groups_mgr.image_groups_tw.child(c_idx),
        range(image_groups_mgr.image_groups_tw.childrenCount())
    )

    ground_truth = []
    pred = []
    for img_group, gt_layer in zip(img_groups_list, gt_layer_list):
        segmentation_channel_group = img_group.getLayersGroup("segmentation")
        pred_layer = segmentation_channel_group.child(0).layer

        ground_truth.append(gt_layer.data[:].flatten())
        pred.append(pred_layer.data[:].flatten())

    return compute_metrics(ground_truth, pred, thresholds=None)


def run_benchmark(input_filenames, ground_truth_filenames, test_filenames,
                  test_ground_truth_filenames):
    viewer = napari.Viewer()
    al_widget = al.get_active_learning_widget()

    train_metrics = run_fine_tuning(input_filenames, ground_truth_filenames,
                                    viewer,
                                    al_widget)

    print(train_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Activelearning benchmarks")

    parser.add_argument("-i", "--input", dest="input_filenames", type=str,
                        nargs="+",
                        help="Input image(s) used to fine-tune a cellpose "
                             "model")
    parser.add_argument("-g", "--ground-truth", dest="ground_truth_filenames",
                        type=str, nargs="+",
                        help="Corresponding ground-truth used to fine-tune the"
                             " model")
    parser.add_argument("-ti", "--test-input", dest="test_input_filenames",
                        type=str, nargs="+",
                        help="Test image(s) used to measure the fine-tuned "
                             "model performance")
    parser.add_argument("-tg", "--test-ground-truth",
                        dest="test_ground_truth_filenames",
                        type=str, nargs="+",
                        help="Corresponding tst ground-truth used to compare "
                             "the fine-tuned model performance")
    args = parser.parse_args()

    run_benchmark(**args.__dict__)
