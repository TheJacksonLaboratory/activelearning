import argparse
from itertools import cycle
import numpy as np
import torch
from cellpose import models, transforms


import datautils
import alutils


def cellpose_model_init():
    model = models.CellposeModel(gpu=False, model_type="cyto")
    model.net.load_model(model.pretrained_model[0], device=model.device)

    model.net.mkldnn = False
    model.net.eval()

    return model


def cellpose_inference(img, model):
    x = transforms.convert_image(img, None, normalize=False, invert=False, nchan=img.shape[-1])
    x = transforms.normalize_img(x, invert=False)

    x = torch.from_numpy(np.moveaxis(x, -1, 0))[None, :2, ...]

    with torch.no_grad():
        y, _ = model(x)
        cellprob = y[0, 2, ...].detach().cpu()
        probs = cellprob.sigmoid().numpy()

    return probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Active learning sample generator")
    parser.add_argument("-i", "--input", dest="train_filenames", nargs="+",
                        type=str,
                        help="Input unlabeled images used for active learning"
                             " sample generation")
    parser.add_argument("-m", "--masks", dest="train_masks_filenames",
                        nargs="+",
                        type=str,
                        help="Masks from input images from where samples can"
                             " be extracted.")
    parser.add_argument("-o", "--output-dir", dest="output_dir", type=str,
                        help="Directory where intermediate outputs are stored",
                        default=".")
    parser.add_argument("-ps", "--patch-size", dest="patch_size", type=int,
                        help="Size of the sampled patches",
                        default=256)
    parser.add_argument("-ns", "--num-samples", dest="samples_per_image",
                        type=int,
                        help="Maximum number of samples to extract from each "
                             "image")
    parser.add_argument("-ms", "--max-samples", dest="max_samples_to_annotate",
                        type=int,
                        help="Maximum number of samples to annotate after all"
                             " samples have been extracted")
    parser.add_argument("-t", "--repetitions", dest="repetitions", type=int,
                        help="Number of repetitions for Montecarlo sampling",
                        default=30)
    parser.add_argument("-w", "--num-workers", dest="num_workers", type=int,
                        help="Number of workers for multi thread data loading",
                        default=0)

    args = parser.parse_args()

    images_metadata = datautils.parse_dataset_metadata(args.train_filenames)

    if args.train_masks_filenames is not None:
        masks_metadata = datautils.parse_dataset_metadata(args.train_masks_filenames)

    else:
        # Mask valid sampling regions in input images
        masks_metadata = []
        for im_metadata in images_metadata:
            layers_masks = datautils.annotate_mask(
                dict(
                    input_image=(im_metadata, True)
                ),
                patch_size=args.patch_size
            )

            mask = layers_masks["input_image"]
            mask_filename = datautils.save_mask(mask, im_metadata["filename"], args.output_dir, mask_data_group="mask")

            masks_metadata.append(dict(
                filenames=mask_filename,
                source_axes="YX",
                data_group="mask/0",
                roi=None
            ))

    output_filenames = datautils.prepare_output_dataset(args.output_dir, images_metadata, args.patch_size)

    cellpose_model = cellpose_model_init()
    # TODO: Generate a DataSpecs object from the confidence maps and output filename 
    active_filenames = alutils.compute_confidence_dataset(cellpose_model.net, cellpose_inference, images_metadata, masks_metadata, output_filenames, patch_size=args.patch_size, samples_per_image=args.samples_per_image, max_samples_to_annotate=args.max_samples_to_annotate, repetitions=args.repetitions, num_workers=args.num_workers)

    for image_metadata, output_filename, _ in active_filenames:
        datautils.downsample_image(output_filename, "confidence_map", num_scales=5)

    alutils.annotate_samples(active_filenames, args.patch_size, args.num_workers)
    for image_metadata, output_filename, _ in active_filenames:
        datautils.downsample_image(output_filename, "annotations", num_scales=5)
