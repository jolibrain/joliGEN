import os
from PIL import Image, ImageDraw, ImageFont

from data.base_dataset import get_transform_ref, get_transform
from data.utils import load_image
from data.unaligned_labeled_mask_online_dataset import UnalignedLabeledMaskOnlineDataset
from data.image_folder import make_ref_path_list
from util.util import tensor2im, add_text2image, im2tensor
import torch
import numpy as np


class UnalignedLabeledMaskOnlinePromptDataset(UnalignedLabeledMaskOnlineDataset):
    def __init__(self, opt, phase, name=""):
        super().__init__(opt, phase, name)

        self.B_img_prompt = make_ref_path_list(self.dir_B, "/prompts.txt")
        self.transform_prompt_img = get_transform(
            self.opt, grayscale=(self.output_nc == 1)
        )

    def get_img(
        self,
        A_img_path,
        A_label_mask_path,
        A_label_cls,
        B_img_path=None,
        B_label_mask_path=None,
        B_label_cls=None,
        index=None,
        clamp_semantics=True,
    ):
        result = super().get_img(
            A_img_path,
            A_label_mask_path,
            A_label_cls,
            B_img_path,
            B_label_mask_path,
            B_label_cls,
            index,
            clamp_semantics,
        )
        img_path_B = result["B_img_paths"]
        real_B_prompt_path = self.B_img_prompt[img_path_B]

        if len(real_B_prompt_path) == 1 and isinstance(real_B_prompt_path[0], str):
            real_B_prompt = real_B_prompt_path[0]

        result.update({"real_B_prompt": real_B_prompt})
        image_numpy_B = tensor2im(result["B"].unsqueeze(0))
        imageB_text = add_text2image(image_numpy_B, real_B_prompt)
        real_B_prompt_img_tensor = im2tensor(imageB_text)

        result.update({"real_B_prompt_img": real_B_prompt_img_tensor})

        return result
