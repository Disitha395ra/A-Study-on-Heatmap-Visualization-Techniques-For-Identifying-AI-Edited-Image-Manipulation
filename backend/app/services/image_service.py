from dataclasses import dataclass
from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from backend.app.config import IMAGE_SIZE


@dataclass
class ProcessedImage:
    tensor: torch.Tensor
    rgb_array: np.ndarray

    original_width: int
    original_height: int

    processed_width: int
    processed_height: int

    original_format: str


class ImageValidationError(ValueError):
    pass


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def process_uploaded_image(image_bytes: bytes) -> ProcessedImage:
    if not image_bytes:
        raise ImageValidationError("The uploaded image is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as opened_image:
            original_format = opened_image.format or "UNKNOWN"

            # Correct images taken from phones/cameras that use EXIF rotation.
            corrected_image = ImageOps.exif_transpose(opened_image)

            original_width, original_height = corrected_image.size

            if original_width <= 0 or original_height <= 0:
                raise ImageValidationError(
                    "The uploaded image has invalid dimensions."
                )

            # Every model was trained with RGB input.
            rgb_image = corrected_image.convert("RGB")

            # Exact training-time preprocessing.
            image_tensor = IMAGE_TRANSFORM(rgb_image).unsqueeze(0)

            # RGB image for later CAM overlay generation.
            resized_rgb = rgb_image.resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                resample=Image.Resampling.BILINEAR,
            )

            rgb_array = (
                np.asarray(resized_rgb, dtype=np.float32) / 255.0
            )

    except UnidentifiedImageError as error:
        raise ImageValidationError(
            "The uploaded file is not a valid supported image."
        ) from error

    except OSError as error:
        raise ImageValidationError(
            "The uploaded image could not be read."
        ) from error

    return ProcessedImage(
        tensor=image_tensor,
        rgb_array=rgb_array,
        original_width=original_width,
        original_height=original_height,
        processed_width=IMAGE_SIZE,
        processed_height=IMAGE_SIZE,
        original_format=original_format,
    )