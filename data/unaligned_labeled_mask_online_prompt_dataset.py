import os
from PIL import Image, ImageDraw, ImageFont

from data.base_dataset import get_transform_ref, get_transform
from data.utils import load_image
from data.unaligned_labeled_mask_online_dataset import UnalignedLabeledMaskOnlineDataset
from data.image_folder import make_ref_path_list
from util.util import tensor2im


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

        # print("real_B_prompt=", real_B_prompt)
        result.update({"real_B_prompt": real_B_prompt})

        image = Image.open(img_path_B)
        draw = ImageDraw.Draw(image)
        font_size = 80
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print("Font not found. Using default font.")
            font = ImageFont.load_default()

        position = (50, 50)  # (x, y) coordinates for the text position
        fill_color = (255, 0, 0)  # White color for the text

        draw.text(position, real_B_prompt, font=font, fill=fill_color)
        output_path = "/data1/juliew/joliGEN/WIP_joliGEN/text_image.png"
        image.save(output_path)
        real_B_prompt_img_tensor = self.transform_prompt_img(image)

        data_B = result["B"]
        image_data_B = tensor2im(data_B)
        image_B = Image.fromarray(image_data_B)
        image_B.save("data_B_dataset.png")

        data_A = result["A"]
        image_data_A = tensor2im(data_A)
        image_A = Image.fromarray(image_data_A)
        image_A.save("data_A_dataset.png")

        result.update({"real_B_prompt_img": real_B_prompt_img_tensor})

        return result
