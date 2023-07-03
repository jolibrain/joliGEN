import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import scipy
import torch
from mobile_sam.modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    TinyViT,
    TwoWayTransformer,
)
from numpy.random import PCG64, Generator
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.utils.transforms import ResizeLongestSide
from torch import nn
from torch.nn import functional as F

from models.modules.utils import download_mobile_sam_weight, download_sam_weight
from util.util import tensor2im


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack(
            [self.preprocess(x["image"]) for x in batched_input], dim=0
        )
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


class SamPredictorG:
    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def set_image(
        self,
        image: torch.Tensor,
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (torch.Tensor): The image already in RGB format
        """

        # Transform the image to the form expected by the model
        # print('image size=', image.size())
        input_image_torch = self.transform.apply_image_torch(image).to(self.device)

        self.set_torch_image(input_image_torch, (image.size(dim=2), image.size(dim=3)))

    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            labels_torch = torch.as_tensor(
                point_labels, dtype=torch.int, device=self.device
            )
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(
                mask_input, dtype=torch.float, device=self.device
            )
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        # masks = masks[0].detach().cpu().numpy()
        # iou_predictions = iou_predictions[0].detach().cpu().numpy()
        # low_res_masks = low_res_masks[0].detach().cpu().numpy()
        # return masks, iou_predictions, low_res_masks
        if not return_logits:
            masks[0] = masks[0].float()
        return masks[0], iou_predictions[0], low_res_masks[0]

    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(
            low_res_masks, self.input_size, self.original_size
        )

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self.features is not None
        ), "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None


class MobileSam(nn.Module):
    """
    The MobileSAM related code has been adapted to our needs from the official
    MobileSAM repository (https://github.com/ChaoningZhang/MobileSAM). Many thanks to
    their team for this great work!
    """

    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: Union[ImageEncoderViT, TinyViT],
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack(
            [self.preprocess(x["image"]) for x in batched_input], dim=0
        )
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


def build_sam_vit_t(checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = MobileSam(
        image_encoder=TinyViT(
            img_size=1024,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    mobile_sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        mobile_sam.load_state_dict(state_dict)
    return mobile_sam


######### JoliGEN level functions
def load_sam_weight(model_path):
    if "vit_h" in model_path:
        model_type = "vit_h"
    elif "vit_l" in model_path:
        model_type = "vit_l"
    elif "vit_b" in model_path:
        model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam_predictor = SamPredictorG(sam)
    return sam, sam_predictor


def load_mobile_sam_weight(model_path):
    sam = build_sam_vit_t(checkpoint=model_path)
    sam_predictor = SamPredictorG(sam)
    return sam, sam_predictor


def predict_sam(img, sam_predictor, bbox=None):
    # - img in RBG value space
    img = torch.clamp(img, min=-1.0, max=1.0)
    img = (img + 1) / 2.0 * 255.0

    # - set image to model
    sam_predictor.set_image(img)

    # - generate keypoints/bbox
    point_coord = []
    if bbox is None:
        prompt_bbox = np.array(
            [0, 0, img.size(dim=2), img.size(dim=3)]
        )  # bbox over the full image
        point_coord = None
        point_label = None
    else:
        prompt_bbox = np.array(bbox)

        # XXX: using center point doesn't yield better results
        #      based on current tests, especially on
        #      non-convex shapes
        #      Deactivated at code level for now
        # point_coord = np.array([[int(prompt_bbox[0][0]+(prompt_bbox[0][2]-prompt_bbox[0][0])/2),
        #                         int(prompt_bbox[0][1]+(prompt_bbox[0][3]-prompt_bbox[0][1])/2)]])

        point_label = np.array([1])
        point_coord = None

    # - get masks as tensors
    masks, scores, _ = sam_predictor.predict(
        point_coords=point_coord,
        point_labels=point_label,
        box=prompt_bbox,
        multimask_output=True,
        return_logits=True,
    )  # in BxCxHxW format, where C is the number of masks

    # - get best mask
    best_mask = None
    best_score = 0.0
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if best_mask == None or best_score < score:
            best_mask = mask
            best_score = score

    return best_mask.unsqueeze(0)


def show_mask(mask, cat):
    # convert true/false mask to cat/0 array
    cat_mask = np.zeros_like(mask)
    cat_mask[mask] = cat
    return cat_mask.astype(np.uint8)


def predict_sam_mask(img, bbox, predictor, cat=1):
    """
    Generate mask from bounding box
    :param img: image tensor(Size[3, H, W])
    :param bbox: bounding box np.array([x1, y1, x2, y2])
    :return: mask
    """

    if img.shape[0] <= 4:
        cv_img = tensor2im(img.unsqueeze(0))
    else:
        cv_img = img
    predictor.set_image(cv_img)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bbox[None, :],
        multimask_output=True,
    )  # outputs a boolean mask (True/False)
    mask = show_mask(masks[np.argmax(scores)], cat)

    return mask


def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def non_max_suppression(masks, threshold):
    selected_indices = []
    for i in range(len(masks)):
        mask_i = masks[i]
        overlap = False
        for j in selected_indices:
            mask_j = masks[j]
            if iou(mask_i, mask_j) > threshold:
                overlap = True
                break
        if not overlap:
            selected_indices.append(i)
    selected_masks = [masks[i] for i in selected_indices]
    return selected_masks


def random_sample_in_circle(n, img_size=128):
    random_generator = Generator(PCG64())
    uniform = random_generator.uniform
    A = []
    x0 = img_size / 2
    y0 = img_size / 2
    radius = img_size / 2
    for _ in range(n):
        x = int(uniform(0, img_size + 1))
        y = int(uniform(0, img_size + 1))
        while (x - x0) ** 2 + (y - y0) ** 2 > radius**2:
            x = int(uniform(0, img_size + 1))
            y = int(uniform(0, img_size + 1))
        A.append((x, y))
    return A


def random_sample_in_ellipse(n, width, height):
    random_generator = Generator(PCG64())
    uniform = random_generator.uniform
    A = []
    x0 = width // 2
    y0 = height // 2
    if x0 == 0:
        x0 = 10e-4
    if y0 == 0:
        y0 = 10e-4
    theta = np.pi / 4
    for _ in range(n):
        x = int(uniform(0, width))
        y = int(uniform(0, height))
        while ((x - x0) ** 2) / (x0**2) * np.sin(2 * theta) ** 2 + ((y - y0) ** 2) / (
            y0**2
        ) * np.sin(2 * theta) ** 2 + (2 * np.cos(2 * theta) * (x - x0) * (y - y0)) / (
            x0 * y0 * np.sin(2 * theta) ** 2
        ) > 1:
            x = int(uniform(0, width))
            y = int(uniform(0, height))
        A.append([x, y])
    return A


def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)
    return image.permute(2, 0, 1).contiguous()


def predict_sam_edges(
    image,
    sam,
    use_gaussian_filter=False,
    use_sobel_filter=False,
    output_binary_sam=False,
    redundancy_threshold=0.88,
    sobel_threshold=0.0,
    gaussian_sigma=3.0,
    final_canny=True,
    canny_threshold1=50,
    canny_threshold2=300,
    min_mask_area=0.001,
    max_mask_area=0.99,
    points_per_side=16,
    sample_points_in_ellipse=True,
):
    """
    Performs edge detection based on SAM predicted masks.

    Arguments:
        image ([np.ndarray]): Batch of image to calculate masks from. Expects
            images in HWC uint8 format, with pixel values in [0, 255].
        sam (Sam): The model to use for mask prediction.
        use_gaussian_filter (bool): Whether to smooth each mask with gaussian blur
            before computing its edges.
        use_sobel_filter (bool): Whether to use a sobel filter on each mask.
        output_binary_sam (bool): Whether to output the sketchified version of the
            image as a binary image or with the original image colors (before Canny).
        redundancy_threshold (float): Threshold for Non-Maximum Suppression.
            A mask sharing redundancy_threshold * 100 % or more of its area with
            another one is not kept.
        sobel_threshold (float): Threshold for the % of gradient magintude to kept
            after Sobel filter.
        gaussian_sigma (float): Standard deviation used to perform Gaussian blur.
        final_canny (bool): Whether to perform a Canny edge detection on
            sam output to soften the edges.
        canny_threshold1 (int): Canny minimum threshold.
        canny_threshold2 (int): Canny maximum threshold.
        min_mask_area (float): Minimum area for a mask to be used, in proportion of the
            image.
        max_mask_area (float): Maximum area for a mask to be used, in proportion of the
            image.
        points_per_side (int): Number of points to use for creating the grid of points
            we prompt Sam (points_per_side * points_per_side points will be prompted).
        sample_points_in_ellipse (bool): Whether to sample the prompted points in an
            ellipse to avoid points in the image corner.

    Returns:
        (np.ndarray): The sketchified (binary) version of the input image in HxW format,
            where (H, W) is the original image size.
    """

    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    batched_input = []
    for k in range(len(image)):
        if sample_points_in_ellipse:
            points = random_sample_in_ellipse(
                points_per_side * points_per_side,
                image[k].shape[1],
                image[k].shape[0],
            )
        else:
            points = [
                [
                    i * image[k].shape[0] // points_per_side,
                    j * image[k].shape[1] // points_per_side,
                ]
                for i in range(points_per_side)
                for j in range(points_per_side)
            ]
        image_points = torch.tensor([[point] for point in points], device=sam.device)

        batched_input.append(
            {
                "image": prepare_image(image[k], resize_transform, sam),
                "point_coords": resize_transform.apply_coords_torch(
                    image_points, image[k].shape[:2]
                ),
                "original_size": image[k].shape[:2],
                "point_labels": torch.ones((points_per_side * points_per_side, 1)),
            }
        )

    batched_output = []
    with torch.no_grad():
        for batch in batched_input:
            batched_output.append(sam([batch], multimask_output=True)[0])

    for k in range(len(image)):
        flat_masks = []
        flat_scores = []
        for mask_out in batched_output[k]["masks"]:
            for mask in mask_out:
                flat_masks.append(mask.cpu().numpy())
        for score_out in batched_output[k]["iou_predictions"]:
            for score in score_out:
                flat_scores.append(score.cpu().numpy())

        flat_masks = np.array(flat_masks)
        flat_scores = np.array(flat_scores)
        sorted_indices = np.argsort(flat_scores)[::-1]
        sorted_masks = flat_masks[sorted_indices]
        sorted_scores = flat_scores[sorted_indices]
        batched_output[k]["sorted_masks"] = sorted_masks
        batched_output[k]["sorted_scores"] = sorted_scores

    for k in range(len(image)):
        non_redundant_masks = non_max_suppression(
            batched_output[k]["sorted_masks"], redundancy_threshold
        )
        non_redundant_masks = np.array(non_redundant_masks)
        batched_output[k]["non_redundant_masks"] = non_redundant_masks

    batched_edges = []
    for k in range(len(image)):
        masked_imgs = []
        for mask in batched_output[k]["non_redundant_masks"]:
            assert (
                mask.shape == image[k].shape[:2]
            ), "mask should be the same size of image"
            prob_map = mask.astype(np.float32)

            if use_gaussian_filter:
                # Apply Gaussian filter to probability map
                sigma = gaussian_sigma  # adjust sigma to control amount of smoothing
                prob_map = scipy.ndimage.gaussian_filter(prob_map, sigma=sigma)
            if use_sobel_filter:
                # Apply Sobel filter to probability map
                sobel_x = cv2.Sobel(prob_map, cv2.CV_32F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(prob_map, cv2.CV_32F, 0, 1, ksize=3)

                # Compute gradient magnitude
                grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

                # set threshold at x% of max gradient magnitude
                threshold = sobel_threshold * np.max(grad_mag)

                edge_map = (grad_mag > threshold).astype(np.uint8)
            else:
                edge_map = (prob_map * 255).astype(np.uint8)
            # Find contours of mask
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) > 0:
                # Find outermost contour using convex hull
                hull = cv2.convexHull(contours[0])

                # Create binary mask of outer boundary pixels
                boundary_mask = np.zeros_like(mask, dtype=np.uint8)
                cv2.drawContours(boundary_mask, [hull], -1, 255, -1)

                # Set values outside boundary to zero
                edge_map[~np.logical_and(edge_map, boundary_mask)] = 0

                # Threshold edge map to create binary mask
                threshold = 0.01
                _, binary_map = cv2.threshold(
                    edge_map, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                binary_map = binary_map.astype(np.uint8)

                if output_binary_sam:
                    masked_img = cv2.cvtColor(binary_map, cv2.COLOR_GRAY2RGB)
                else:
                    # Apply binary mask to original input image
                    masked_img = cv2.bitwise_and(image[k], image[k], mask=binary_map)
                if len(masked_img[masked_img > 0]) > 0:
                    # take masks for which area is greater than min_mask_area % of the image and less than max_mask_area % of the image
                    if (
                        len(masked_img[masked_img > 0])
                        >= min_mask_area * masked_img.shape[0] * masked_img.shape[1]
                        and len(masked_img[masked_img > 0])
                        <= max_mask_area * masked_img.shape[0] * masked_img.shape[1]
                    ):
                        masked_imgs.append(masked_img)

        if len(masked_imgs) > 0:
            masked_imgs = np.array(masked_imgs)
            # Take pixel-wise max over all masked images
            final_pred = np.max(masked_imgs, axis=0)
            # Linearly normalize final prediction to range [0, 1]
            normalized_pred = (final_pred - np.min(final_pred)) / (
                np.max(final_pred) - np.min(final_pred)
            )
            normalized_pred = (normalized_pred * 255).astype(np.uint8)

            # Apply edge nms (=Canny) to thicken edges
            threshold1 = min(canny_threshold1, canny_threshold2)
            threshold2 = max(canny_threshold1, canny_threshold2)
            if final_canny:
                edges = cv2.Canny(normalized_pred, threshold1, threshold2)

                batched_edges.append(edges)
            else:
                batched_edges.append(cv2.cvtColor(normalized_pred, cv2.COLOR_BGR2GRAY))
        else:
            batched_edges.append(np.zeros_like(image[k][:, :, 0]))
    return batched_edges


def compute_mask_with_sam(img, rect_mask, sam_model, device, batched=True):
    # get bbox and cat from rect_mask

    if not batched:
        indices = torch.nonzero(rect_mask.squeeze(0))
        if indices.numel() != 0:
            masks_exist = True
            x_min = indices[:, 1].min()
            y_min = indices[:, 0].min()
            x_max = indices[:, 1].max()
            y_max = indices[:, 0].max()
            box = torch.tensor([x_min, y_min, x_max, y_max])
            category = int(torch.unique(rect_mask).max())
        else:
            masks_exist = False
            box = torch.tensor([0, 0, 0, 0])
            category = 0

        box = box.to(device)

        if masks_exist:
            mask = predict_sam_mask(
                img=img,
                bbox=np.array(box.cpu()),
                predictor=SamPredictor(sam_model),
                cat=category,
            )
            sam_masks = torch.from_numpy(mask).to(device)
        else:
            sam_masks = rect_mask
        return sam_masks
    else:
        boxes = torch.zeros((rect_mask.shape[0], 4))
        categories = []
        masks_exist = []

        for i in range(rect_mask.shape[0]):
            mask = rect_mask[i].squeeze()
            indices = torch.nonzero(mask)
            if indices.numel() != 0:
                masks_exist.append(True)
                x_min = int(indices[:, 1].min())
                y_min = int(indices[:, 0].min())
                x_max = int(indices[:, 1].max())
                y_max = int(indices[:, 0].max())
                boxes[i] = torch.tensor([x_min, y_min, x_max, y_max])
                categories.append(int(torch.unique(mask).max()))
            else:
                masks_exist.append(False)
                boxes[i] = torch.tensor([0, 0, 0, 0])
                categories.append(0)

        boxes = boxes.to(device)
        sam_masks = torch.zeros_like(rect_mask)
        predictor = SamPredictor(sam_model)
        for i in range(rect_mask.shape[0]):
            if masks_exist[i]:
                mask = predict_sam_mask(
                    img=img[i],
                    bbox=np.array([int(coord) for coord in boxes[i].cpu()]),
                    predictor=predictor,
                    cat=categories[i],
                )
                sam_masks[i] = torch.from_numpy(mask).unsqueeze(0)
            else:
                sam_masks[i] = rect_mask[i]
        return sam_masks


def init_sam_net(model_type_sam, model_path, device):
    if model_type_sam == "sam":
        download_sam_weight(path=model_path)
        freezenet_sam, predictor_sam = load_sam_weight(model_path=model_path)
        if device is not None:
            freezenet_sam = freezenet_sam.to(device)
    elif model_type_sam == "mobile_sam":
        download_mobile_sam_weight(path=model_path)
        freezenet_sam, predictor_sam = load_mobile_sam_weight(model_path=model_path)
        if device is not None:
            freezenet_sam.to(device)
    else:
        raise ValueError(
            f'{model_type_sam} is not a correct choice for model_type_sam.\nChoices: ["sam", "mobile_sam"]'
        )
    return freezenet_sam, predictor_sam
