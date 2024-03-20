from data.unaligned_labeled_mask_dataset import UnalignedLabeledMaskDataset
import torch


class UnalignedLabeledMaskClsDataset(UnalignedLabeledMaskDataset):
    def __init__(self, opt, phase, name=""):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        UnalignedLabeledMaskDataset.__init__(self, opt, phase, name)

        self.A_label_cls = []
        self.B_label_cls = []

        for label in self.A_label:
            label_split = label.split(" ")
            assert len(label_split) == 2
            self.A_label_cls.append(label_split[0])

        if self.use_domain_B and hasattr(self, "B_label"):
            for label in self.B_label:
                label_split = label.split(" ")
                assert len(label_split) == 2
                self.B_label_cls.append(label_split[0])

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

        # TODO : check how to deal with float for regression
        return_dict["A_label_cls"] = torch.tensor(int(A_label_cls))
        if B_label_cls is not None:
            return_dict["B_label_cls"] = torch.tensor(int(B_label_cls))

        return return_dict
