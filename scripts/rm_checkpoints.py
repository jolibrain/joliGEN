import argparse
import os


def main():
    main_parser = argparse.ArgumentParser()

    main_parser.add_argument(
        "--checkpoint_dir", type=str, default="", help="path to checkpoints directory."
    )

    main_parser.add_argument(
        "--preserved_epoch",
        type=str,
        default=[],
        nargs="*",
        help="iterations numbers that we want to preserve models weights",
    )

    main_parser.add_argument(
        "--preserved_iter",
        type=str,
        default=[],
        nargs="*",
        help="epochs numbers that we want to preserve models weights",
    )

    main_parser.add_argument("--verbose", action="store_true")

    opt = main_parser.parse_args()

    list_files = os.listdir(opt.checkpoint_dir)

    file_types = [".pt", ".pth", ".onnx"]

    preserved_epoch = opt.preserved_epoch

    preserved_iter = ["iter_" + cur_iter for cur_iter in opt.preserved_iter]

    preserved_steps = preserved_epoch + preserved_iter + ["latest"]

    for cur_file in list_files:
        cur_file_type = os.path.splitext(cur_file)[1]
        step = cur_file.split("_")[0]

        if step == "iter":
            step = "_".join(cur_file.split("/")[-1].split("_")[:2])

        if cur_file_type in file_types and step not in preserved_steps:
            os.remove(os.path.join(opt.checkpoint_dir, cur_file))
            if opt.verbose:
                print("%s was removed." % cur_file)
        else:
            if opt.verbose:
                print("%s was kept." % cur_file)


if __name__ == "__main__":
    main()
