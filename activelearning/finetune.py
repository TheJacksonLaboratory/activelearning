import argparse
import numpy as np
import torch


import datautils
import alutils

from cellpose import models, transforms


def cellpose_model_init():
    model = models.CellposeModel(gpu=False, model_type="cyto")
    model.net.load_model(model.pretrained_model[0], device=model.device)

    model.net.mkldnn = False
    model.net.eval()

    return model


def cellpose_inference(img, model):
    x = transforms.convert_image(img, None, normalize=False, invert=False,
                                 nchan=img.shape[-1])
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
    dataset_metadata = list(dict(images=im_md) for im_md in images_metadata)

    if args.train_masks_filenames is not None:
        masks_metadata_list = \
            datautils.parse_dataset_metadata(args.train_masks_filenames)
        for image_metadata_dict, mask_metadata in zip(dataset_metadata,
                                                      masks_metadata_list):
            image_metadata_dict["masks"] = mask_metadata

    else:
        # Mask valid sampling regions in input images
        for image_metadata_dict in dataset_metadata:
            datautils.annotate_mask(
                args.output_dir,
                image_metadata_dict,
                layer_to_annotate="images",
                annotation_name="sampling_mask",
                scale=1 / args.patch_size,
                mask_modality="masks",
                mask_dtype=np.int64
            )

    cellpose_model = cellpose_model_init()

    active_dataset_metadata = alutils.compute_confidence_dataset(
        cellpose_model.net,
        cellpose_inference,
        args.output_dir,
        dataset_metadata,
        patch_size=args.patch_size,
        samples_per_image=args.samples_per_image,
        max_samples_to_annotate=args.max_samples_to_annotate,
        repetitions=args.repetitions,
        num_workers=args.num_workers
    )

    datautils.downsample_image(
        **image_metadata_dict["confidence_maps"],
        num_cales=5
    )

    for image_metadata in active_dataset_metadata:
        # Remove the superpixel modality
        image_metadata.pop("superpixels")

    for image_metadata_dict in active_dataset_metadata:
        datautils.annotate_mask(args.output_dir, image_metadata_dict,
                                layer_to_annotate="confidence_maps",
                                annotation_name="annotations",
                                patch_size=args.patch_size,
                                mask_dtype=np.int32,
                                mask_modality="labels")

    datautils.downsample_image(
        **image_metadata_dict["annotations"],
        num_cales=5
    )
