from data.unaligned_labeled_mask_online_dataset import UnalignedLabeledMaskOnlineDataset


class UnalignedLabeledMaskClsOnlineDataset(UnalignedLabeledMaskOnlineDataset):
    def __init__(self, opt, phase):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        UnalignedLabeledMaskOnlineDataset.__init__(self, opt, phase)
        self.header = ["img", "cls", "mask"]

    def get_img(
        self,
        A_img_path,
        A_label_mask_path,
        A_label_cls,
        B_img_path=None,
        B_label_mask_path=None,
        B_label_cls=None,
        index=None,
    ):
        return_dict = super().get_img(
            A_img_path,
            A_label_mask_path,
            A_label_cls,
            B_img_path,
            B_label_mask_path,
            B_label_cls,
            index,
        )

        if return_dict is None:
            return None

        return_dict["A_label_cls"] = self.cat_A_ref_bbox
        if hasattr(self, "B_label_cls"):
            return_dict["B_label_cls"] = self.cat_B_ref_bbox

        return return_dict
