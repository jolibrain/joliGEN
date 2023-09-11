import os

from torchvision.transforms.functional import resize
from PIL import Image

from data.base_dataset import get_transform_ref
from data.unaligned_labeled_mask_dataset import UnalignedLabeledMaskDataset
from data.image_folder import make_ref_path


class UnalignedLabeledMaskRefDataset(UnalignedLabeledMaskDataset):
    def __init__(self, opt, phase):
        super().__init__(opt, phase)

        self.A_img_ref = make_ref_path(self.dir_A, "/conditions.txt")

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

        ref_A_path = self.A_img_ref[img_path]

        if self.opt.data_relative_paths:
            ref_A_path = os.path.join(self.root, ref_A_path)

        try:
            ref_A = Image.open(ref_A_path).convert("RGB")

        except Exception as e:
            print(
                "failure with reading A domain image ref ",
                ref_A_path,
            )
            print(e)
            return None

        ref_A = self.transform_ref(ref_A)

        result.update({"ref_A": ref_A})

        return result
