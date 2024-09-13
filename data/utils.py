from PIL import Image


def load_image(img_path, img_bits=8, use_tiff=False):
    if use_tiff:
        img = tifffile.imread(img_path)
    else:
        img = Image.open(img_path)

    if img_bits == 8:
        img = img.convert("RGB")

    return img


# def load_image(img_path):
#     image = Image.open(img_path)

#     # Define the color for transparency (e.g., transparent black)
#     transparent_black = (0, 0, 0, 0)

#     # Convert the image to RGBA mode if needed
#     image = image.convert("RGBA")

#     # Create a new image with the specified color for transparency
#     transparent_color = Image.new("RGBA", image.size, transparent_black)

#     # Use alpha_composite to make the specified color transparent
#     result = Image.alpha_composite(transparent_color, image)

#     # Convert the result back to RGB mode
#     result_rgb = result.convert("RGB")

#     return result_rgb
