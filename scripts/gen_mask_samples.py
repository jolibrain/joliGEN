import argparse
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, "../")
sys.path.insert(0, "./")
from scripts.gen_single_image_diffusion import generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-in-file",
        help="file path to generator model (.pth file)",
        required=True,
    )
    parser.add_argument(
        "--model-norm", action="store_true", help="whether the model is normalized"
    )

    parser.add_argument("--img-width", default=-1, type=int, help="image width")
    parser.add_argument("--img-height", default=-1, type=int, help="image height")

    parser.add_argument(
        "--crop-width", default=-1, type=int, help="crop width (optional)"
    )
    parser.add_argument(
        "--crop-height", default=-1, type=int, help="crop height (optional)"
    )

    # parser.add_argument("--img-in", help="image to transform", required=True)
    # parser.add_argument(
    #    "--mask-in", help="mask used for image transformation", required=False
    # )
    # parser.add_argument("--bbox-in", help="bbox file used for masking")
    # parser.add_argument("--img-out", help="transformed image", required=True)
    parser.add_argument(
        "--mask-in", help="mask used for image transformation", required=False
    )
    parser.add_argument("--dir-out", help="output directory", required=True)
    parser.add_argument(
        "--data-in", help="data input txt file with image and bbox files", required=True
    )
    parser.add_argument(
        "--bbox-use", default=-1, type=int, help="bbox id to sample from (default all)"
    )
    parser.add_argument(
        "--sampling-steps", default=-1, type=int, help="number of sampling steps"
    )
    parser.add_argument("--cpu", action="store_true", help="whether to use CPU")
    parser.add_argument("--gpuid", type=int, default=0, help="which GPU to use")
    parser.add_argument(
        "--seed", type=int, default=-1, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--nimgs", default=1, type=int, help="number of different images to iterate"
    )
    parser.add_argument(
        "--nsamples", default=1, type=int, help="number of samples per image"
    )

    parser.add_argument(
        "--transfer",
        action="store_true",
        help="whether to transfer conditioning from another image",
    )
    parser.add_argument(
        "--source-img",
        default="PanneauToulouse.png",
        help="image to use as conditionment",
    )
    parser.add_argument(
        "--source-bbox",
        default=None,
        help="bbox file for image tu use as conditionment",
    )
    parser.add_argument(
        "--source-id", type=int, default=0, help="line of bbox file tu use"
    )

    parser.add_argument(
        "--create-collage",
        action="store_true",
        help="whether to create a collage to compare generations",
    )
    args = parser.parse_args()

    ##- iterate files and generate multiple samples per file
    lines = open(args.data_in, "r").read().splitlines()
    logs = []
    ## collage image
    img = np.zeros((1, 1, 3), np.uint8)
    # sublines = random.sample(lines, 3)
    for img_nb, line in enumerate(lines):
        ##- sample multiple outputs
        for i in range(args.nimgs):
            elts = line.rstrip().split()
            with open(elts[1], "r") as f:
                bbox_count = len(f.readlines())
            imgname = os.path.basename(elts[0])
            ##- generate for each bbox
            for j in range(0, bbox_count):
                # for k in range(0, args.nsamples):
                args.img_in = elts[0]
                args.bbox_in = elts[1]
                args.img_out = os.path.join(
                    args.dir_out, imgname.replace(".jpg", "") + "-" + str(j) + ".jpg"
                )
                args.bbox_width_factor = 0
                args.bbox_height_factor = 0
                args.write = True
                args.mask_square = True
                img_in, _ = os.path.splitext(args.img_in)
                args.name = os.path.basename(img_in) + "_" + str(i) + "_" + str(j)
                args.mask_delta = [0]
                args.previous_frame = None
                args.bbox_id = j
                ##- save orig, mask, out
                if args.bbox_use >= 0 and args.bbox_use < bbox_count:
                    if j != args.bbox_use:
                        continue
                    else:
                        try:
                            generate(**vars(args))
                        except Exception as e:
                            print(e)
                            logs.append(str(img_nb) + str(i) + str(j))
                            logs.append(str(e))
                else:
                    try:
                        generate(**vars(args))
                    except Exception as e:
                        print(e)
                        logs.append(str(img_nb) + str(i) + str(j))
                        logs.append(str(e))

                if args.create_collage:
                    extensions = ("_y_0.jpg", "_cond.jpg", "_generated_crop.jpg")
                    img_paths = [
                        os.path.join(args.dir_out, args.name + ext)
                        for ext in extensions
                    ]
                    for i in range(len(img_paths)):
                        image = cv2.imread(img_paths[i])
                        if i % 3 == 0:
                            row = image
                        else:
                            row = cv2.hconcat([row, image])

                        if i == len(img_paths) - 1 or i % 3 == 2:
                            if img.shape[0] == 1:
                                img = row
                            else:
                                img = cv2.vconcat([img, row])

                            # reset the row
                            row = np.zeros((1, 1, 3), np.uint8)
    if args.create_collage:
        cv2.imwrite("Collage_output.jpg", img)

    with open("logs.txt", "w") as logfile:
        logfile.writelines(logs)
