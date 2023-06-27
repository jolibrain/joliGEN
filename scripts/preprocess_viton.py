import argparse
from pathlib import Path
import zipfile
from tqdm import tqdm
import os
import shutil
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("VITON-HD dataset preprocessing")
    parser.add_argument(
        "--zip-file",
        type=str,
        required=True,
        help="path to the VITON-HD zip file",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="path to an output folder for preprocessed data",
    )
    parser.add_argument(
        "--dilate",
        type=int,
        default=1,
        help="size of the square kernel for mask dilation",
    )
    return parser.parse_args()


def process(image, zf, target_dir, dilate):
    stage = Path("trainA" if "train/" in image else "testA")
    basename = Path(image).stem

    # extract raw image
    rel_image = stage / "imgs" / (basename + ".jpg")
    target = target_dir / rel_image
    target.write_bytes(zf.read(image))

    # extract mask
    mask = image.replace("/image/", "/image-parse-v3/").replace(".jpg", ".png")
    mask = zf.read(mask)
    mask = cv2.imdecode(np.frombuffer(mask, np.uint8), 1)
    orange = np.array([0, 85, 254])
    mask = cv2.inRange(mask, orange, orange)
    mask = np.clip(mask, 0, 1)
    kernel = np.ones((dilate, dilate), np.uint8)
    mask = cv2.dilate(mask, kernel)
    rel_mask = stage / "mask" / (basename + ".png")
    target = target_dir / rel_mask
    cv2.imwrite(str(target), mask)

    # add paths
    pairs = target_dir / stage / "paths.txt"
    pairs = pairs.open("a")
    pairs.write(f"{rel_image} {rel_mask}\n")


def main():
    args = parse_args()

    # create dataset folders
    zip_file = Path(args.zip_file)
    assert zip_file.is_file()
    target_dir = Path(args.target_dir)
    assert not target_dir.exists()
    for folder1 in ["trainA", "testA"]:
        for folder2 in ["imgs", "mask"]:
            folder = target_dir / folder1 / folder2
            folder.mkdir(parents=True)

    # process images
    zf = zipfile.ZipFile(zip_file)
    images = [name for name in zf.namelist() if "/image/" in name and "_00.jpg" in name]
    for image in tqdm(images):
        process(image, zf, target_dir, args.dilate)


if __name__ == "__main__":
    main()
