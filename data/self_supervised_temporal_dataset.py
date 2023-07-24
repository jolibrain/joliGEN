import torch

from data.temporal_labeled_mask_online_dataset import TemporalLabeledMaskOnlineDataset
from data.online_creation import fill_mask_with_random, fill_mask_with_color


class SelfSupervisedTemporalDataset(TemporalLabeledMaskOnlineDataset):
    """
    This dataset class can create datasets with mask labels from one domain.
    """

    def __init__(self, opt, phase):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt, phase)

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
        result = super().get_img(
            A_img_path,
            A_label_mask_path,
            A_label_cls,
            B_img_path,
            B_label_mask_path,
            B_label_cls,
            index,
        )

        try:
            A_img_list = [result["A"][0]]
            if self.opt.data_online_creation_rand_mask_A:
                A_img = fill_mask_with_random(
                    result["A"][1], result["A_label_mask"][1], -1
                )
            elif self.opt.data_online_creation_color_mask_A:
                A_img = fill_mask_with_color(
                    result["A"][1], result["A_label_mask"][1], {}
                )
            else:
                raise Exception(
                    "self supervised dataset: no self supervised method specified"
                )

            A_img_list.append(A_img)

            A_img_list = torch.stack(A_img_list)

            result.update(
                {
                    "A": A_img_list,
                    "B": result["A"],
                    "B_label_mask": result["A_label_mask"].clone(),
                }
            )
        except Exception as e:
            print(e, "self supervised data loading")
            return None

        return result
