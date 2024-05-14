import os
from PIL import Image

from data.base_dataset import get_transform_ref
from data.utils import load_image
from data.unaligned_labeled_mask_online_dataset import UnalignedLabeledMaskOnlineDataset
from data.image_folder import make_ref_path_list


class UnalignedLabeledMaskOnlineRefDataset(UnalignedLabeledMaskOnlineDataset):
    def __init__(self, opt, phase, name=""):
        super().__init__(opt, phase, name)

        self.B_img_ref_prompt = make_ref_path_list(self.dir_B, "/prompt.txt")


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

        img_path = result["A_img_paths"]

        if self.opt.data_relative_paths:
            img_path = img_path.replace(self.root, "")


        ref_B_prompt_path = self.B_img_ref_prompt[img_path]

        if self.opt.data_relative_paths:
            ref_B_prompt_path = os.path.join(self.root, ref_B_prompt_path)

        try:
            with open(ref_B_prompt_path, "r") as file:
                ref_B_prompt = file.read()

        except Exception as e:
            print(
                "failure with reading B domain prompt ref ",
                ref_B_prompt_path,
            )
            print(e)
            return None

        result.update({"ref_B_prompt": ref_B_prompt})

        return result
