"""Generates a joliGEN dataset from BDD100K."""
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-label-file", type=Path, help="path to JSON BDD100k label file"
    )
    parser.add_argument(
        "--path-to-imgs", type=Path, help="path to BDD100k image directory"
    )
    parser.add_argument(
        "--bbox-out-dir", type=Path, help="output directory for bbox dataset files"
    )
    parser.add_argument(
        "--time-of-day", action="store_true", help="whether to segment per time of day"
    )
    parser.add_argument(
        "--weather", action="store_true", help="whether to segment per weather type"
    )
    parser.add_argument(
        "--force-time-of-day",
        choices=["", "daytime", "night"],
        default="",
        help="force filter per time of day, e.g. to remove night",
    )

    args = parser.parse_args()

    if not args.time_of_day and not args.weather:
        print("use --time-of-day or --weather")
        sys.exit()

    return args


def load_label_file(json_label_path: Path) -> list[dict]:
    with open(json_label_path, "r") as json_label_file:
        data = json.load(json_label_file)

    print(f"Successfully loaded label file {json_label_path}.")
    return data


def parse_bboxes(sample: dict) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Parses all bboxes in the given sample.
    Returns an empty list if no labels or if no box2d present.

    All bboxes are represented by a tuple: (category, (xmin, xmax, ymin, ymax)).
    """
    bboxes = list()
    if "labels" not in sample:
        return []

    for label in sample["labels"]:
        if "box2d" not in label:
            continue

        box2d = label["box2d"]
        xmin = int(box2d["x1"])
        xmax = int(box2d["x2"])
        ymin = int(box2d["y1"])
        ymax = int(box2d["y2"])

        bboxes.append((label["category"], (xmin, xmax, ymin, ymax)))

    return bboxes


def parse_label_file(
    data: list[dict],
    segment_weather: bool,
    segment_time_of_day: bool,
    force_time_of_day: str,
) -> tuple[dict[str, list], set[str], set[str]]:
    """Read the whole data and return the list of bboxes for each image along with
    the set of all categories and segments.

    ---
    Args:
        data: List of samples (a dict) containing the BDD100k label data.
        segment_weather: Whether to segment by weather.
        segment_time_of_day: Whether to segment by time of day.
        force_time_of_day: If not empty, only include samples with this time of day.

    ---
    Returns:
        A tuple containing:
        - A dict mapping image names to a list of bboxes.
        - A set of all categories.
        - A set of all segments.
    """
    assert (
        segment_weather or segment_time_of_day
    ), "Must segment by weather or time of day."

    bbox_files = dict()
    categories = set()
    segments = set()

    for sample in tqdm(data):
        image_name = sample["name"]
        metadata = sample["attributes"]

        if force_time_of_day != "" and metadata["timeofday"] != force_time_of_day:
            continue

        bboxes = parse_bboxes(sample)
        if bboxes == []:
            continue

        segment = (
            metadata["weather"]
            if segment_weather
            else metadata["timeofday"].replace("/", "_")
        )
        segments.add(segment)

        bbox_files[(image_name, segment)] = bboxes
        categories |= {category for category, _ in bboxes}

    return bbox_files, categories, segments


if __name__ == "__main__":
    args = parse_args()

    print(f"Bboxes will be located in {args.bbox_out_dir}.")
    args.bbox_out_dir.mkdir(parents=True, exist_ok=False)

    data = load_label_file(args.json_label_file)

    bbox_files, categories, segments = parse_label_file(
        data, args.weather, args.time_of_day, args.force_time_of_day
    )

    print(f"\n{len(categories)} categories:")
    category_to_id = {cat: id for id, cat in enumerate(sorted(categories))}
    for category, id in category_to_id.items():
        print(f"{id}: {category}")

    print(f"\n{len(segments)} segments:")
    segment_to_id = {seg: id for id, seg in enumerate(sorted(segments))}
    for segment, id in segment_to_id.items():
        print(f"{id}: {segment}")

    segment_paths = {segment: [] for segment in segments}
    for (image_name, segment), bboxes in tqdm(
        bbox_files.items(), desc="Writing bbox files"
    ):
        image_path = args.path_to_imgs / image_name
        bbox_path = (args.bbox_out_dir / image_name).with_suffix(".txt")

        # Absolute value is easier to work with, you can juste move the paths.txt files as you wish.
        # Comment those two lines if you want to work with relative paths.
        image_path = image_path.absolute()
        bbox_path = bbox_path.absolute()

        segment_paths[segment].append(f"{image_path} {bbox_path}")

        # Read the bboxes of the given image and save them in a single bbox file.
        # The bbox file is formatted such that each line is:
        # category_id xmin xmax ymin ymax
        bbox_lines = [
            [category_to_id[category], *coords] for category, coords in bboxes
        ]
        bbox_lines = [" ".join(str(value) for value in line) for line in bbox_lines]
        with open(bbox_path, "w") as bbox_file:
            bbox_file.write("\n".join(bbox_lines))

    for segment, paths in tqdm(
        segment_paths.items(), desc="Writing final paths.txt file"
    ):
        filename = f"bdd100k_{segment}.txt"
        with open(filename, "w") as path_file:
            path_file.write("\n".join(paths))
