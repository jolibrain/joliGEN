import cv2
import re
from tqdm import tqdm
import os

from gen_single_image_diffusion import generate
from diffusion_options import DiffusionOptions


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split("(\d+)", text)]


if __name__ == "__main__":

    options = DiffusionOptions()

    options.parser.add_argument("--dataroot", help="images to transform", required=True)

    options.parser.add_argument(
        "--video-width", default=-1, type=int, help="video width"
    )

    options.parser.add_argument(
        "--video-height", default=-1, type=int, help="video height"
    )

    options.parser.add_argument("--fps", default=30, type=int, help="video fps")

    options.parser.add_argument(
        "--nb_img_max", default=10000, type=int, help="max number of images compute"
    )

    options.parser.add_argument(
        "--sv_frames", action="store_true", help="whether to save frames"
    )

    options.parser.add_argument(
        "--cond",
        type=str,
        help="whether to save frames",
        choices=["previous", "zero", "generated"],
    )

    options.parser.add_argument(
        "--start_frame",
        default=-1,
        type=int,
        help="if >0, resume at the corresponding frame",
    )

    args = options.parse()

    args.img_out = None
    args.mask_in = None
    args.bbox_in = None
    args.write = args.sv_frames

    real_name = args.name

    if args.video_width == -1:
        args.video_width = args.img_width

    if args.video_height == -1:
        args.video_width = args.img_width

    with open(args.dataroot, "r") as f:
        paths_list = f.read().split("\n")

    images = []
    labels = []

    for line in paths_list:
        line_split = line.split(" ")

        if len(line_split) == 2:
            images.append(line_split[0])
            labels.append(line_split[1])

    images.sort(key=natural_keys)
    labels.sort(key=natural_keys)

    if args.start_frame != -1:
        images, labels = images[args.start_frame :], labels[args.start_frame :]

    images, labels = images[: args.nb_img_max], labels[: args.nb_img_max]

    out = cv2.VideoWriter(
        os.path.join(args.dir_out, args.name + "_generated_video.avi"),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),  # "H", "2", "6", "4"
        #  "m", "p", "4", "v"
        args.fps,
        (args.video_width, args.video_height),
    )

    assert out.isOpened()

    args.previous_frame = None
    args.previous_frame_bbox_in = None

    for i, (image, label) in tqdm(enumerate(zip(images, labels)), total=len(images)):

        args.img_in = image

        if label.endswith(".txt"):
            args.bbox_in = label
        else:
            args.mask_in = label

        if args.write:
            args.name = (
                real_name
                + "_"
                + str(i).zfill(
                    len(str(len(images)))
                )  # .zfill(len(str(args.nb_samples)))
            )  # zfill(len(str(images)))

        """if not args.use_real_previous and i == 0:
            args.write = False"""

        frame = generate(**vars(args))

        args.previous_frame_bbox_in = label

        if args.cond == "previous":  # use_real_previous:
            args.previous_frame = image
        elif args.cond == "generated":
            if i == 0:
                args.previous_frame = image
                # args.write = True
                continue
            args.previous_frame = frame
        elif args.cond == "zero":
            if i == 0:
                if reconstruction_guidance:
                    args.previous_frame = frame
                else:
                    args.previous_frame = None

        colored_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        out.write(frame)

    # When everything done, release the video write objects
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()
