import shutil
from itertools import product
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid


def make_grid(image: np.ndarray, thresholds: tuple, n_steps: int) -> plt.Figure:
    # Make a grid of the product of the two linspace based on the thresholds range.
    # The grid is a 2D array of shape (n_steps, n_steps, 2)
    thresholds_grid = np.array(
        np.meshgrid(
            np.linspace(thresholds[0], thresholds[1], n_steps, dtype=np.int32),
            np.linspace(thresholds[0], thresholds[1], n_steps, dtype=np.int32),
        )
    ).T

    fig = plt.figure(figsize=(18.0, 20.0))
    image_grid = ImageGrid(
        fig, 111, nrows_ncols=(n_steps + 1, n_steps), axes_pad=0.1, label_mode="all"
    )

    for x, y in product(
        range(thresholds_grid.shape[0]), range(thresholds_grid.shape[1])
    ):
        t1, t2 = thresholds_grid[x, y]
        canny = cv2.Canny(image, t1, t2)
        canny = cv2.cvtColor(canny, cv2.COLOR_BGR2RGB)

        coords = (y + 1) * n_steps + x

        image_grid[coords].imshow(canny)

        # Show xlabels.
        if x == 0:
            image_grid[coords].set_ylabel(f"{t2}")

        # Show ylabels.
        if y + 1 == n_steps:
            image_grid[coords].set_xlabel(f"{t1}")

    # Remove ticks.
    for grid_plot in image_grid:
        grid_plot.set_xticks([])
        grid_plot.set_yticks([])

        grid_plot.set_xticklabels([])
        grid_plot.set_yticklabels([])

    # The image is plotted in the upper-left corner.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_grid[0].imshow(image)

    return fig


if __name__ == "__main__":
    import argparse
    import random

    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        help="Directory containing images to transform",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save transformed images",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--threshold1",
        help="First threshold for canny edge detection",
        type=int,
        default=000,
    )
    parser.add_argument(
        "--threshold2",
        help="Second threshold for canny edge detection",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--n_steps",
        help="Number of steps for the linspace",
        type=int,
        default=11,
    )
    parser.add_argument(
        "--samples",
        help="Number of samples to take from the image directory",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--size",
        help="Size of the image",
        type=int,
        default=64,
    )
    args = parser.parse_args()

    assert (
        args.threshold1 < args.threshold2
    ), "Threshold 1 must be smaller than threshold 2"

    random.seed(42)
    images = shutil.os.listdir(args.image_dir)
    images = images[: args.samples]

    args.output_dir.mkdir(exist_ok=True)

    for image_name in tqdm(images, desc="Grid canny"):
        image = cv2.imread(str(args.image_dir / image_name))
        image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_CUBIC)

        fig = make_grid(image, (args.threshold1, args.threshold2), args.n_steps)

        canny_path = (args.output_dir / image_name).with_suffix(".png")
        fig.savefig(canny_path, bbox_inches="tight")
