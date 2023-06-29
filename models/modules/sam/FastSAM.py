import os

import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from util.util import tensor2im

from ..utils import download_fastsam_weight


def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def segment_image(image, bbox):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    x1, y1, x2, y2 = bbox
    segmented_image_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (255, 255, 255))
    # transparency_mask = np.zeros_like((), dtype=np.uint8)
    transparency_mask = np.zeros(
        (image_array.shape[0], image_array.shape[1]), dtype=np.uint8
    )
    transparency_mask[y1:y2, x1:x2] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode="L")
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image


def format_results(result, filter=0):
    annotations = []
    n = len(result.masks.data)
    for i in range(n):
        annotation = {}
        mask = result.masks.data[i] == 1.0

        if torch.sum(mask) < filter:
            continue
        annotation["id"] = i
        annotation["segmentation"] = mask.cpu().numpy()
        annotation["bbox"] = result.boxes.data[i]
        annotation["score"] = result.boxes.conf[i]
        annotation["area"] = annotation["segmentation"].sum()
        annotations.append(annotation)
    return annotations


def filter_masks(annotations):  # filte the overlap mask
    annotations.sort(key=lambda x: x["area"], reverse=True)
    to_remove = set()
    for i in range(0, len(annotations)):
        a = annotations[i]
        for j in range(i + 1, len(annotations)):
            b = annotations[j]
            if i != j and j not in to_remove:
                # check if
                if b["area"] < a["area"]:
                    if (a["segmentation"] & b["segmentation"]).sum() / b[
                        "segmentation"
                    ].sum() > 0.8:
                        to_remove.add(j)

    return [a for i, a in enumerate(annotations) if i not in to_remove], to_remove


def get_bbox_from_mask(mask):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    x1, y1, w, h = cv2.boundingRect(contours[0])
    x2, y2 = x1 + w, y1 + h
    if len(contours) > 1:
        for b in contours:
            x_t, y_t, w_t, h_t = cv2.boundingRect(b)
            x1 = min(x1, x_t)
            y1 = min(y1, y_t)
            x2 = max(x2, x_t + w_t)
            y2 = max(y2, y_t + h_t)
        h = y2 - y1
        w = x2 - x1
    return [x1, y1, x2, y2]


# clip
@torch.no_grad()
def retrieve(model, preprocess, elements: [Image.Image], search_text: str, device):
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100.0 * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)


def crop_image(annotations, image):
    image = Image.fromarray(image)
    ori_w, ori_h = image.size
    mask_h, mask_w = annotations[0]["segmentation"].shape
    if ori_w != mask_w or ori_h != mask_h:
        image = image.resize((mask_w, mask_h))
    cropped_boxes = []
    cropped_images = []
    not_crop = []
    filter_id = []
    # annotations, _ = filter_masks(annotations)
    # filter_id = list(_)
    for _, mask in enumerate(annotations):
        if np.sum(mask["segmentation"]) <= 100:
            filter_id.append(_)
            continue
        bbox = get_bbox_from_mask(mask["segmentation"])
        cropped_boxes.append(segment_image(image, bbox))
        cropped_images.append(bbox)

    return cropped_boxes, cropped_images, not_crop, filter_id, annotations


def compute_box_prompt(masks, bbox, target_height, target_width):
    h = masks.shape[1]
    w = masks.shape[2]
    if h != target_height or w != target_width:
        bbox = [
            int(bbox[0] * w / target_width),
            int(bbox[1] * h / target_height),
            int(bbox[2] * w / target_width),
            int(bbox[3] * h / target_height),
        ]
    bbox[0] = round(bbox[0]) if round(bbox[0]) > 0 else 0
    bbox[1] = round(bbox[1]) if round(bbox[1]) > 0 else 0
    bbox[2] = round(bbox[2]) if round(bbox[2]) < w else w
    bbox[3] = round(bbox[3]) if round(bbox[3]) < h else h

    # IoUs = torch.zeros(len(masks), dtype=torch.float32)
    bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

    masks_area = torch.sum(masks[:, bbox[1] : bbox[3], bbox[0] : bbox[2]], dim=(1, 2))
    orig_masks_area = torch.sum(masks, dim=(1, 2))

    union = bbox_area + orig_masks_area - masks_area
    IoUs = masks_area / union
    max_iou_index = torch.argmax(IoUs)

    return masks[max_iou_index].cpu().numpy(), max_iou_index


def compute_point_prompt(masks, points, pointlabel, target_height, target_width):
    h = masks[0]["segmentation"].shape[0]
    w = masks[0]["segmentation"].shape[1]
    if h != target_height or w != target_width:
        points = [
            [int(point[0] * w / target_width), int(point[1] * h / target_height)]
            for point in points
        ]
    onemask = np.zeros((h, w))
    for i, annotation in enumerate(masks):
        if type(annotation) == dict:
            mask = annotation["segmentation"]
        else:
            mask = annotation
        for i, point in enumerate(points):
            if mask[point[1], point[0]] == 1 and pointlabel[i] == 1:
                onemask += mask
            if mask[point[1], point[0]] == 1 and pointlabel[i] == 0:
                onemask -= mask
    onemask = onemask >= 1
    return onemask, 0


def compute_text_prompt(annotations, img, device, text_prompt):
    cropped_boxes, _, _, filter_id, annotations = crop_image(annotations, img)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    scores = retrieve(clip_model, preprocess, cropped_boxes, text_prompt, device=device)
    max_idx = scores.argsort()
    max_idx = max_idx[-1]
    max_idx += sum(np.array(filter_id) <= int(max_idx))
    return annotations[max_idx]["segmentation"], max_idx


def load_fastSam_weight(model_path):
    if os.path.exists(model_path):
        sam_model = YOLO(model_path)
        sam_model.parameters = sam_model.model.parameters
        return sam_model
    else:
        download_fastsam_weight(model_path)
        sam_model = YOLO(model_path)
        sam_model.parameters = sam_model.model.parameters
        return sam_model


def fastSAM(
    fastsam_model,
    img,
    imgsz=1024,
    iou=0.9,
    text_prompt=None,
    box_prompt=[0, 0, 0, 0],
    conf=0.4,
    point_prompt=[[0, 0]],
    point_label=[0],
    device="cuda:0",
    retina=True,
):
    ### TODO: Gérer les batchs d'images
    if torch.is_tensor(img):
        img = torch.clamp(img, min=-1.0, max=1.0)
        img = (img + 1) / 2.0 * 255.0
        img = tensor2im(img)
    results = fastsam_model(
        img,
        imgsz=imgsz,
        device=device,
        retina_masks=retina,
        iou=iou,
        conf=conf,
        max_det=300,
        verbose=False,
    )
    if box_prompt[2] != 0 and box_prompt[3] != 0:
        annotations = prompt(
            results,
            img,
            text_prompt,
            box_prompt,
            point_prompt,
            point_label,
            device,
            box=True,
        )
        annotations = np.array([annotations])
    elif text_prompt != None:
        results = format_results(results[0], 0)
        annotations = prompt(
            results,
            img,
            text_prompt,
            box_prompt,
            point_prompt,
            point_label,
            device,
            text=True,
        )
        annotations = np.array([annotations])
    elif point_prompt[0] != [0, 0]:
        results = format_results(results[0], 0)
        annotations = prompt(
            results,
            img,
            text_prompt,
            box_prompt,
            point_prompt,
            point_label,
            device,
            point=True,
        )
        # list to numpy
        annotations = np.array([annotations])
    else:
        annotations = results[0].masks.data

    return annotations


def prompt(
    results,
    img,
    text_prompt,
    box_prompt,
    point_prompt,
    point_label,
    device,
    box=None,
    point=None,
    text=None,
):
    ori_img = img.copy()
    ori_h = ori_img.shape[0]
    ori_w = ori_img.shape[1]
    if results[0].masks is None:
        return [None]
    if box:
        mask, _ = compute_box_prompt(
            results[0].masks.data,
            convert_box_xywh_to_xyxy(box_prompt),
            ori_h,
            ori_w,
        )
    elif point:
        mask, _ = compute_point_prompt(results, point_prompt, point_label, ori_h, ori_w)
    elif text:
        mask, _ = compute_text_prompt(results, img, device, text_prompt)
    else:
        return None
    return mask
