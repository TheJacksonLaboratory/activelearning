import os
import argparse
import napari
import napari_activelearning as al

from cellpose.metrics import average_precision, aggregated_jaccard_index


def compute_metrics(ground_truth, pred):
    ap, tp, fp, fn = average_precision(ground_truth, pred, threshold=[0.5])
    f1_score = 2 * tp / (2 * tp + fp + fn)

    aji = aggregated_jaccard_index(ground_truth, pred)

    return dict(f1_score=f1_score, avg_prec=ap, aggregated_jaccard=aji,
                tp=tp, fp=fp, fn=fn)


def run_inference(input_filenames, ground_truth_filenames, viewer, al_widget):
    image_groups_mgr = al_widget.widget(0)
    acquisition_fun_mgr = al_widget.widget(1)
    min_size = {"Y": float("inf"), "X": float("inf")}

    in_layer_list = []
    gt_layer_list = []
    img_groups_list = []

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
        img_groups_list.append(image_groups_mgr._active_image_group)

        image_groups_mgr._active_image_group.child(0).setSelected(True)
        image_groups_mgr.image_groups_editor.edit_axes_le.setText("YXC")
        image_groups_mgr.image_groups_editor.update_source_axes()

        image_groups_mgr._active_image_group.child(1).setSelected(True)
        image_groups_mgr.image_groups_editor.use_as_labels_chk.setChecked(True)

        min_size["Y"] = min(in_layer_list[-1].data.shape[0], min_size["Y"])
        min_size["X"] = min(in_layer_list[-1].data.shape[1], min_size["X"])

    acquisition_fun_mgr.patch_sizes_mspn.sizes = {"Y": min_size["Y"],
                                                  "X": min_size["X"]}
    acquisition_fun_mgr.max_samples_spn.setValue(2)

    _ = acquisition_fun_mgr.compute_acquisition_layers(run_all=True)

    ground_truth = []
    pred = []
    for img_group, gt_layer in zip(img_groups_list, gt_layer_list):
        segmentation_channel_group = img_group.getLayersGroup("segmentation")
        pred_layer = segmentation_channel_group.child(0).layer

        ground_truth.append(gt_layer.data[:])
        pred.append(pred_layer.data[0].compute())

    return compute_metrics(ground_truth, pred)


def clear_layers_list(viewer):
    layers_list = list(viewer.layers)
    for layer in layers_list:
        viewer.layers.remove(layer)


def save_benchmarks(output_filename, input_filenames, experiment_identifier,
                    data_mode,
                    metrics_dict):
    new_file = not os.path.exists(output_filename)
    with open(output_filename, "w" if new_file else "a") as fp_out:
        if new_file:
            fp_out.write("input_filename,experiment,mode,f1_score,avg_prec,"
                         "aggregated_jaccard,tp,fp,fn\n")

        for in_fn, f1_score, avg_prec, aji, tp, fp, fn in zip(
            input_filenames,
            metrics_dict["f1_score"].flatten(),
            metrics_dict["avg_prec"].flatten(),
            metrics_dict["aggregated_jaccard"].flatten(),
            metrics_dict["tp"].flatten(),
            metrics_dict["fp"].flatten(),
            metrics_dict["fn"].flatten()
        ):
            fp_out.write(f"{in_fn},{experiment_identifier},{data_mode},"
                         f"{f1_score},{avg_prec},{aji},{tp},{fp},{fn}\n")


def run_benchmark(input_filenames, ground_truth_filenames,
                  test_input_filenames,
                  test_ground_truth_filenames,
                  output_filename):
    viewer = napari.Viewer()
    al_widget = al.get_active_learning_widget()
    acquisition_fun_mgr = al_widget.widget(1)

    acquisition_fun_mgr.methods_cmb.setCurrentIndex(2)
    acquisition_fun_mgr.tunable_segmentation_method\
                       ._segmentation_parameters\
                       .channels.value = (0, 1)
    acquisition_fun_mgr.tunable_segmentation_method\
                       ._segmentation_parameters\
                       .model_type.value = "cyto3"

    test_metrics = run_inference(test_input_filenames,
                                 test_ground_truth_filenames,
                                 viewer,
                                 al_widget)
    save_benchmarks(output_filename, test_input_filenames, "baseline",
                    "test", test_metrics)
    clear_layers_list(viewer)

    train_metrics = run_inference(input_filenames, ground_truth_filenames,
                                  viewer,
                                  al_widget)
    save_benchmarks(output_filename, input_filenames, "baseline", "train",
                    train_metrics)

    acquisition_fun_mgr.tunable_segmentation_method\
                       ._finetuning_parameters\
                       .model_name.value = "cyto3_ft"
    acquisition_fun_mgr.tunable_segmentation_method\
                       ._finetuning_parameters\
                       .batch_size.value = min(8,
                                               int(0.8 * len(input_filenames)))
    acquisition_fun_mgr.tunable_segmentation_method\
                       ._finetuning_parameters\
                       .learning_rate.value = 0.0001
    acquisition_fun_mgr.tunable_segmentation_method\
                       ._finetuning_parameters\
                       .n_epochs.value = 20

    _ = acquisition_fun_mgr.fine_tune()
    clear_layers_list(viewer)

    train_metrics = run_inference(input_filenames, ground_truth_filenames,
                                  viewer,
                                  al_widget)
    save_benchmarks(output_filename, input_filenames, "finetuned", "train",
                    train_metrics)
    clear_layers_list(viewer)

    test_metrics = run_inference(test_input_filenames,
                                 test_ground_truth_filenames,
                                 viewer,
                                 al_widget)
    save_benchmarks(output_filename, test_input_filenames, "finetuned", "test",
                    test_metrics)


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
    parser.add_argument("-o", "--output",
                        dest="output_filename",
                        type=str,
                        help="Filename where to store the experiment results",
                        default="metrics_finetuning.csv")
    args = parser.parse_args()

    run_benchmark(**args.__dict__)
