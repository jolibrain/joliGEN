import os
from PIL import Image

from data.base_dataset import get_transform_ref
from data.utils import load_image
from data.unaligned_labeled_mask_online_dataset import UnalignedLabeledMaskOnlineDataset
from data.image_folder import make_ref_path_list


class UnalignedLabeledMaskOnlineRefDataset(UnalignedLabeledMaskOnlineDataset):
    def __init__(self, opt, phase):
        super().__init__(opt, phase)

        self.A_img_ref = make_ref_path_list(self.dir_A, "/conditions.txt")

        self.transform_ref = get_transform_ref(opt)

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

        A_ref_bbox_id = result["A_ref_bbox_id"]

        ref_A_path = self.A_img_ref[img_path][A_ref_bbox_id]

        if self.opt.data_relative_paths:
            ref_A_path = os.path.join(self.root, ref_A_path)

        try:
            ref_A = load_image(ref_A_path)

        except Exception as e:
            print(
                "failure with reading A domain image ref ",
                ref_A_path,
            )
            print(e)
            return None

        ref_A_norm = self.transform_ref(ref_A)
        result.update({"ref_A": ref_A_norm})

        return result
