import os

from PIL import Image

from data.base_dataset import get_transform_ref
from data.utils import load_image
from data.unaligned_labeled_mask_online_dataset import UnalignedLabeledMaskOnlineDataset
from data.image_folder import make_ref_path_list
 


class UnalignedLabeledMaskOnlinePromptDataset(UnalignedLabeledMaskOnlineDataset):
    def __init__(self, opt, phase, name=""):
        super().__init__(opt, phase, name)

        self.B_img_prompt = make_ref_path_list(self.dir_B, "/prompts.txt")


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

        # print()
        img_path_B = result["B_img_paths"]

        real_B_prompt_path = self.B_img_prompt[img_path_B]

        if len(real_B_prompt_path) == 1 and isinstance(real_B_prompt_path[0], str):
            real_B_prompt = real_B_prompt_path[0]


        # print("real_B_prompt=", real_B_prompt)
        result.update({"real_B_prompt": real_B_prompt})

        return result
