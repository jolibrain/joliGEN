import os
import sys
import urllib.request

import cv2
import numpy as np
import torch
from basicsr.utils.download_util import load_file_from_url
from einops import rearrange
from torch.nn import functional as F
from torchvision.transforms import Grayscale

from models.mbv2_mlsd_large import MobileV2_MLSD_Large
from models.modules.utils import download_midas_weight, predict_depth

sys.path.append("./../")


def deccode_output_score_and_ptss(tpMap, topk_n=200, ksize=5):
    """
    tpMap:
    center: tpMap[1, 0, :, :]
    displacement: tpMap[1, 1:5, :, :]
    """
    b, c, h, w = tpMap.shape
    assert b == 1, "only support bsize==1"
    displacement = tpMap[:, 1:5, :, :][0]
    center = tpMap[:, 0, :, :]
    heat = torch.sigmoid(center)
    hmax = F.max_pool2d(heat, (ksize, ksize), stride=1, padding=(ksize - 1) // 2)
    keep = (hmax == heat).float()
    heat = heat * keep
    heat = heat.reshape(
        -1,
    )

    scores, indices = torch.topk(heat, topk_n, dim=-1, largest=True)
    yy = torch.floor_divide(indices, w).unsqueeze(-1)
    xx = torch.fmod(indices, w).unsqueeze(-1)
    ptss = torch.cat((yy, xx), dim=-1)

    ptss = ptss.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    displacement = displacement.detach().cpu().numpy()
    displacement = displacement.transpose((1, 2, 0))
    return ptss, scores, displacement


def pred_lines(image, model, input_shape=[256, 256], score_thr=0.10, dist_thr=20.0):
    h, w, _ = image.shape
    h_ratio, w_ratio = [h / input_shape[0], w / input_shape[1]]
    resized_image = np.concatenate(
        [
            cv2.resize(
                image, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA
            ),
            np.ones([input_shape[0], input_shape[1], 1]),
        ],
        axis=-1,
    )

    resized_image = resized_image.transpose((2, 0, 1))
    batch_image = np.expand_dims(resized_image, axis=0).astype("float32")
    batch_image = (batch_image / 127.5) - 1.0

    batch_image = torch.from_numpy(batch_image).float().cuda()
    outputs = model(batch_image)
    pts, pts_score, vmap = deccode_output_score_and_ptss(outputs, 200, 3)
    start = vmap[:, :, :2]
    end = vmap[:, :, 2:]
    dist_map = np.sqrt(np.sum((start - end) ** 2, axis=-1))

    segments_list = []
    for center, score in zip(pts, pts_score):
        y, x = center
        distance = dist_map[y, x]
        if score > score_thr and distance > dist_thr:
            disp_x_start, disp_y_start, disp_x_end, disp_y_end = vmap[y, x, :]
            x_start = x + disp_x_start
            y_start = y + disp_y_start
            x_end = x + disp_x_end
            y_end = y + disp_y_end
            segments_list.append([x_start, y_start, x_end, y_end])

    lines = 2 * np.array(segments_list)  # 256 > 512
    lines[:, 0] = lines[:, 0] * w_ratio
    lines[:, 1] = lines[:, 1] * h_ratio
    lines[:, 2] = lines[:, 2] * w_ratio
    lines[:, 3] = lines[:, 3] * h_ratio

    return lines


class MLSDdetector:
    def __init__(self):
        dir = os.path.dirname(__file__)
        model_dir = os.path.join(dir, "../models/configs/pretrained/")
        remote_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth"
        model_path = os.path.join(model_dir, "mlsd_large_512_fp32.pth")
        if not os.path.exists(model_path):
            load_file_from_url(remote_model_path, model_dir=model_dir)

        model = MobileV2_MLSD_Large()
        model.load_state_dict(torch.load(model_path), strict=True)
        self.model = model.cuda().eval()

    def __call__(self, input_image, thr_v, thr_d):
        assert input_image.ndim == 3
        img = input_image
        img_output = np.zeros_like(img)
        img_um = cv2.UMat(img_output)
        try:
            with torch.no_grad():
                lines = pred_lines(
                    img, self.model, [img.shape[0], img.shape[1]], thr_v, thr_d
                )
                for line in lines:
                    x_start, y_start, x_end, y_end = [int(val) for val in line]
                    cv2.line(
                        img_um,
                        (x_start, y_start),
                        (x_end, y_end),
                        [255, 255, 255],
                        1,
                    )

        except Exception as e:
            print(e)
            pass

        img_with_line = img_um.get()
        return img_with_line[:, :, 0]


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


def fill_img_with_canny(img, mask, low_threshold=150, high_threshold=200):
    img_orig = img.clone()
    mask_2D = mask.cpu()[0, :, :][0]  # Convert mask to a 2D array
    coords = np.column_stack(
        np.where(mask_2D > 0.9)
    )  # Get the coordinates of the white pixels in the mask
    x_0, y_0, w, h = cv2.boundingRect(coords.astype(int))
    to_sketch = img_orig[:, :, x_0 : x_0 + w, y_0 : y_0 + h]

    to_sketch = np.transpose(to_sketch.squeeze().cpu().numpy(), (1, 2, 0))
    edges = cv2.Canny((to_sketch * 255).astype(np.uint8), low_threshold, high_threshold)

    # edges = np.transpose(edges, (2, 0, 1))
    edges = torch.from_numpy(edges).unsqueeze(0).unsqueeze(0)
    img_orig[:, :, x_0 : x_0 + w, y_0 : y_0 + h] = edges

    return img_orig


def fill_img_with_hed(img, mask):
    apply_hed = HEDdetector()
    img_orig = img.clone()
    mask_2D = mask.cpu()[0, :, :][0]  # Convert mask to a 2D array
    coords = np.column_stack(np.where(mask_2D > 0.9))
    x_0, y_0, w, h = cv2.boundingRect(coords.astype(int))
    to_sketch = img_orig[:, :, x_0 : x_0 + w, y_0 : y_0 + h]

    to_sketch = np.transpose(to_sketch.squeeze().cpu().numpy(), (1, 2, 0))
    to_sketch = (to_sketch * 255).astype(np.uint8)
    detected_map = apply_hed(to_sketch) / 255
    detected_map_resized = torch.from_numpy(detected_map).unsqueeze(0).unsqueeze(0)
    img_orig[:, :, x_0 : x_0 + w, y_0 : y_0 + h] = detected_map_resized

    return img_orig


def fill_img_with_hed_Caffe(img, mask):
    """
    From pretrained Caffe model with openCV.
    """

    img_orig = img.clone()
    mask_2D = mask.cpu()[0, :, :][0]  # Convert mask to a 2D array
    coords = np.column_stack(np.where(mask_2D > 0.9))
    x_0, y_0, w, h = cv2.boundingRect(coords.astype(int))

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
    dir = os.path.dirname(__file__)
    folder = os.path.join(dir, "../models/configs/pretrained/")

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
    hed = (hed).astype("uint8")

    hed = torch.from_numpy(hed).unsqueeze(0).unsqueeze(0)

    img_orig[:, :, x_0 : x_0 + w, y_0 : y_0 + h] = hed

    return img_orig


def fill_img_with_hough(
    img, mask, value_threshold=1e-05, distance_threshold=10.0, with_canny=False
):
    if with_canny:
        img_orig = fill_img_with_canny(img, mask)
    else:
        img_orig = img.clone()
    mask_2D = mask.cpu()[0, :, :][0]  # Convert mask to a 2D array
    coords = np.column_stack(np.where(mask_2D > 0.9))
    x_0, y_0, w, h = cv2.boundingRect(coords.astype(int))
    ## TODO check if [:, :, w, h] or invert w and h ?
    to_sketch = img_orig.clone()
    to_sketch = np.transpose(to_sketch.squeeze().cpu().numpy(), (1, 2, 0))
    to_sketch = (to_sketch * 255).astype(np.uint8)
    to_sketch = to_sketch
    apply_mlsd = MLSDdetector()
    detected_map = apply_mlsd(
        to_sketch, thr_v=value_threshold, thr_d=distance_threshold
    )
    detected_map = detected_map / 255
    detected_map_resized = torch.from_numpy(detected_map).unsqueeze(0).unsqueeze(0)
    img_orig[:, :, x_0 : x_0 + w, y_0 : y_0 + h] = detected_map_resized[
        :, :, x_0 : x_0 + w, y_0 : y_0 + h
    ]

    return img_orig


def fill_img_with_depth(img, mask, depth_network="DPT_SwinV2_T_256"):
    img_orig = img.clone()
    mask_2D = mask.cpu()[0, :, :][0]  # Convert mask to a 2D array
    coords = np.column_stack(
        np.where(mask_2D > 0.9)
    )  # Get the coordinates of the white pixels in the mask
    x_0, y_0, w, h = cv2.boundingRect(coords.astype(int))
    to_sketch = img_orig.clone()
    # to_sketch = np.transpose(to_sketch.squeeze().cpu().numpy(), (1, 2, 0))
    midas_w = download_midas_weight(model_type=depth_network)
    depth_map = (
        predict_depth(img=to_sketch, midas=midas_w, model_type=depth_network) / 255
    )
    depth_map = depth_map.unsqueeze(0)
    img_orig[:, :, x_0 : x_0 + w, y_0 : y_0 + h] = depth_map[
        :, :, x_0 : x_0 + w, y_0 : y_0 + h
    ]

    return img_orig


if __name__ == "__main__":
    img = cv2.imread(
        "/data3/killian/mapillary/tlse79/4pAOUUhR5UkZqGOWlf07AA_2_2_y_0.jpg"
    )

    tensor_image = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

    tensor_image = tensor_image / 255.0
    mask = cv2.imread(
        "/data3/killian/mapillary/tlse79/4pAOUUhR5UkZqGOWlf07AA_2_2_mask.jpg"
    )
    tensor_mask = torch.from_numpy(np.transpose(mask, (2, 0, 1))).float()
    tensor_mask = tensor_mask / 255.0
    img_detect = fill_img_with_hough(
        tensor_image.unsqueeze(0), tensor_mask.unsqueeze(0)
    )
