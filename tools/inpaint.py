import os
import numpy as np
import PIL
from PIL import Image, ImageFilter, ImageOps

from .debug import DebugDumper

def blend_images(image1, image2, transition_mask):
    """
    Blend two PIL images using a grayscale mask.

    :param image1: First input image (PIL Image).
    :param image2: Second input image (PIL Image).
    :param mask: Grayscale mask to determine blending (PIL Image).
    :return: Blended image (PIL Image).
    """

    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    parent_dir = os.path.dirname(current_script_dir)
    parent_dir = os.path.join(parent_dir, 'output_images')

    debugDumper = DebugDumper.GetByName("nm_inpainter", parent_dir)

    if image1.size != image2.size or image1.size != transition_mask.size or image1.size != transition_mask.size:
        raise ValueError("All images must be the same size")

    # Convert PIL images to NumPy arrays
    image1_np = np.array(image1)
    image2_np = np.array(image2)
    transition_mask_np = np.array(transition_mask) / 255.0

    debugDumper.dump_image('transition_mask_np', transition_mask_np, level=5)

    # Check if the images have alpha channel; if so, use only the color channels
    if image1_np.shape[2] == 4:
        image1_np = image1_np[..., :3]
    if image2_np.shape[2] == 4:
        image2_np = image2_np[..., :3]

    # Ensure mask is compatible for broadcasting
    if len(transition_mask_np.shape) == 2:  # If the mask is single channel
        transition_mask_np = transition_mask_np[:, :, None]

    # Blend images
    blended_np = image1_np * transition_mask_np + image2_np * (1 - transition_mask_np)

    debugDumper.dump_image('blended_np', blended_np, level=5)

    # Convert the blended array back to a PIL Image
    blended_image = PIL.Image.fromarray(np.uint8(blended_np))

    return blended_image

def normalize_pil(image: Image):

    image_np = np.array(image)

    if len(image_np.shape) > 2:
        if image_np.shape[2] == 4:
            image_np = image_np[..., :3]

    min = image_np.min()
    max = image_np.max()

    if max == min:
        return None
    else:
        image_np = (image_np - min) / (max - min)

    result_image = Image.fromarray((image_np*255).astype(np.uint8))

    return result_image

def nm_inpaint(image: Image, input_mask: Image):

    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    parent_dir = os.path.dirname(current_script_dir)
    parent_dir = os.path.join(parent_dir, 'output_images')

    debugDumper = DebugDumper.GetByName("nm_inpainter", parent_dir)

    # Check the image mode
    if input_mask.mode == 'RGB' or input_mask.mode == 'RGBA':
        splitted_mask = input_mask.split()
        internal_mask = splitted_mask[0].convert("L")
        external_mask = splitted_mask[1].convert("L")

    else:
        print(f"NeuralMaster Inpainter: Input mask does not have RGB channels, 'Original' mode will used")
        return image, input_mask

    debugDumper.dump_image('image', image, level=5)
    debugDumper.dump_image('input_mask', input_mask, level=5)

    internal_mask = normalize_pil(internal_mask)
    external_mask = normalize_pil(external_mask)

    debugDumper.dump_image('internal_mask', internal_mask, level=5)
    debugDumper.dump_image('external_mask', external_mask, level=5)

    if internal_mask is None:
        print(f"Given inpainting mask is uniform, Original mode will be used")
        return image, input_mask

    if external_mask is None:
        print(f"Given external mask is uniform, Original mode will be used")
        return image, input_mask

    transition_internal_mask = internal_mask.point(lambda x: (x-127)*2 if x >= 128 else 0)
    bool_internal_mask = internal_mask.point(lambda x: 255 if x >= 128 else 0)

    debugDumper.dump_image('transition_internal_mask', transition_internal_mask, level=5)
    debugDumper.dump_image('bool_internal_mask', bool_internal_mask, level=5)

    combined_mask = Image.new("L", bool_internal_mask.size, 0)
    combined_mask.paste(ImageOps.invert(bool_internal_mask), (0, 0), mask=external_mask)

    debugDumper.dump_image('combined_mask', combined_mask.convert("RGBA"), level=5)

    image_mod = Image.new('RGBA', (image.width, image.height))

    image_masked = Image.new('RGBa', (image.width, image.height))
    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=combined_mask)

    image_masked = image_masked.convert('RGBa')

    debugDumper.dump_image('image_masked', image_masked.convert("RGBA"), level=5)

    for radius, repeats in [(256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert('RGBA')
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)

    image_mod = image_mod.convert("RGB")
    debugDumper.dump_image('image_mod', image_mod, level=5)

    if image.size != transition_internal_mask.size:
        transition_internal_mask = transition_internal_mask.resize(image.size)

    image_mod = blend_images(image_mod, image, transition_internal_mask)

    if external_mask is not None:
        image_mod.paste(image, mask=ImageOps.invert(external_mask))

    debugDumper.dump_image('image_mod', image_mod, level=1)
    debugDumper.dump_image('internal_mask', internal_mask, level=1)

    return image_mod, internal_mask
