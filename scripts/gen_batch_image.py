import os
import subprocess
import argparse


def main(args):
    # Read the image-prompt pairs from the prompts.txt file
    with open(args.prompt_file, "r") as file:
        lines = file.readlines()

    image_prompt_pairs = [line.strip().split() for line in lines]

    for img_path, prompt_path in image_prompt_pairs:
        # Read the prompt from the prompt file
        with open(prompt_path, "r") as pf:
            prompt = pf.readline().strip()

        # Construct the output file path
        img_out = os.path.join(args.output_folder, os.path.basename(img_path))

        # Prepare the command to call gen_single_image.py
        cmd = [
            "python3",
            "scripts/gen_single_image.py",
            "--model_in_file",
            args.model_in_file,
            "--img_in",
            img_path,
            "--img_out",
            img_out,
            "--prompt",
            prompt,
            "--gpuid",
            str(args.gpuid),
            "--img_width",
            str(args.img_width),
            "--img_height",
            str(args.img_height),
        ]
        if args.compare:
            cmd.append("--compare")

        # Call gen_single_image.py for the current image
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_in_file", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="File containing the image-prompt pairs",
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="Folder to save output images"
    )
    parser.add_argument("--gpuid", type=int, default=0, help="GPU id to use")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Whether to compare original and generated images",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=256,
        help="image width, defatus to model crop sizes",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=256,
        help="image width, defatus to model crop sizes",
    )

    args = parser.parse_args()

    main(args)
