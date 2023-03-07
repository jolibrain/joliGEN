import os
import urllib.request

import cv2
import numpy as np
import torch
from einops import rearrange
from torchvision.transforms import Grayscale


class Network(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(inplace=False),
        )

        self.netScoreOne = torch.nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreTwo = torch.nn.Conv2d(
            in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreThr = torch.nn.Conv2d(
            in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreFou = torch.nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreFiv = torch.nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0
        )

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0
            ),
            torch.nn.Sigmoid(),
        )

        self.load_state_dict(
            {
                strKey.replace("module", "net"): tenWeight
                for strKey, tenWeight in torch.load(model_path).items()
            }
        )

    def forward(self, tenInput):
        tenInput = tenInput * 255.0
        tenInput = tenInput - torch.tensor(
            data=[104.00698793, 116.66876762, 122.67891434],
            dtype=tenInput.dtype,
            device=tenInput.device,
        ).view(1, 3, 1, 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(
            input=tenScoreOne,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        tenScoreTwo = torch.nn.functional.interpolate(
            input=tenScoreTwo,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        tenScoreThr = torch.nn.functional.interpolate(
            input=tenScoreThr,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        tenScoreFou = torch.nn.functional.interpolate(
            input=tenScoreFou,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        tenScoreFiv = torch.nn.functional.interpolate(
            input=tenScoreFiv,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        return self.netCombine(
            torch.cat(
                [tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv], 1
            )
        )


class HEDdetector:
    def __init__(self):
        dir = os.path.dirname(__file__)
        model_dir = os.path.join(dir, "../models/configs/pretrained/")
        remote_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/network-bsds500.pth"
        modelpath = os.path.join(model_dir, "network-bsds500.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url

            load_file_from_url(remote_model_path, model_dir=model_dir)
        self.netNetwork = Network(modelpath).cuda().eval()

    def __call__(self, input_image):
        assert input_image.ndim == 3
        input_image = input_image[:, :, ::-1].copy()
        with torch.no_grad():
            image_hed = torch.from_numpy(input_image).float().cuda()
            image_hed = image_hed / 255.0
            image_hed = rearrange(image_hed, "h w c -> 1 c h w")
            edge = self.netNetwork(image_hed)[0]
            edge = (edge.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            return edge[0]


def fill_img_with_sketch(img, mask):
    grayscale = Grayscale(3)
    gray = grayscale(img)

    threshold = torch.tensor((120 / 255) * 2 - 1)

    thresh = (gray < threshold) * 1.0  # thresh = ((gray < threshold) * 1.0) * 2 - 1

    mask = torch.clamp(mask, 0, 1)

    return mask * thresh + (1 - mask) * img


def fill_img_with_edges(img, mask):
    device = img.device

    edges_list = []

    for cur_img in img:
        cur_img = (
            (torch.einsum("chw->hwc", cur_img).cpu().numpy() + 1) * 255 / 2
        ).astype(np.uint8)
        edges = cv2.Canny(cur_img, 100, 150)
        edges = (
            (((torch.tensor(edges, device=device) / 255) * 2) - 1)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        edges_list.append(edges)

    edges = torch.cat(edges_list, dim=0)

    mask = torch.clamp(mask, 0, 1)

    return mask * edges + (1 - mask) * img


def fill_img_with_canny(img, mask, low_threshold=250, high_threshold=500):
    img_orig = img.clone()
    mask_2D = mask.cpu()[0, :, :][0]  # Convert mask to a 2D array
    coords = np.column_stack(
        np.where(mask_2D > 0)
    )  # Get the coordinates of the white pixels in the mask
    x_0, y_0, w, h = cv2.boundingRect(coords.astype(np.int))
    ## TODO check if [:, :, w, h] or invert w and h ?
    to_sketch = img_orig[:, :, x_0 : x_0 + w, y_0 : y_0 + h]

    to_sketch = np.transpose(to_sketch.squeeze().cpu().numpy(), (1, 2, 0))
    edges = cv2.Canny((to_sketch * 255).astype(np.uint8), low_threshold, high_threshold)
    # edges = np.transpose(edges, (2, 0, 1))
    edges = torch.from_numpy(edges).unsqueeze(0).unsqueeze(0)
    img_orig[:, :, x_0 : x_0 + w, y_0 : y_0 + h] = edges

    return img_orig


def fill_img_with_hed(img, mask):
    apply_hed = HEDdetector()
    detected_map = apply_hed(img)
    print(detected_map.shape)


def fill_img_with_hed_Caffe(img, mask):
    """
    From pretrained Caffe model with openCV.
    """

    img_orig = img.clone()
    mask_2D = mask.cpu()[0, :, :][0]  # Convert mask to a 2D array
    coords = np.column_stack(np.where(mask_2D > 0))
    # Get the coordinates of the white pixels in the mask
    x_0, y_0, w, h = cv2.boundingRect(coords.astype(np.int))
    to_sketch = img_orig[:, :, x_0 : x_0 + w, y_0 : y_0 + h]

    to_sketch = np.transpose(to_sketch.squeeze().cpu().numpy(), (1, 2, 0))

    img_to_sketch = cv2.cvtColor(to_sketch, cv2.COLOR_RGB2BGR)

    (H, W) = img_to_sketch.shape[:2]

    blob = cv2.dnn.blobFromImage(
        img_to_sketch, scalefactor=1.0, size=(W, H), swapRB=False, crop=False
    )

    proto_url = "https://raw.githubusercontent.com/richzhang/colorization/caffe/models/colorization_deploy_v2.prototxt"
    weights_url = "http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel"

    p_filename = "deploy.protoxt"
    w_filename = "pretrained_hed.caffemodel"
    folder = "../models/modules/caffe"

    p_path = os.path.join(folder, p_filename)
    w_path = os.path.join(folder, w_filename)

    if not os.path.exists(folder):
        os.makedirs(folder)

    # Download file from URL and save it as local file
    if not os.path.exists(p_path):
        urllib.request.urlretrieve(proto_url, p_path)
    if not os.path.exists(w_path):
        urllib.request.urlretrieve(weights_url, w_path)
    ## Load the pretrained Caffe model
    net = cv2.dnn.readNetFromCaffe(p_path, w_path)
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")

    hed = torch.from_numpy(hed).unsqueeze(0).unsqueeze(0)

    img_orig[:, :, x_0 : x_0 + w, y_0 : y_0 + h] = hed

    return img_orig


def fill_img_with_hough(img, mask):
    return None


if __name__ == "__main__":
    img = torch.rand(1, 3, 256, 256).squeeze(0)
    mask = torch.rand(1, 1, 256, 256).squeeze(0)
    fill_img_with_hed(img, mask)
