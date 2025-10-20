
# background_remover.py - Remove backgrounds from clothing images
from rembg import remove
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BackgroundRemover:
    """
    Remove backgrounds from clothing/fashion images
    Supports multiple models for best results
    """

    def __init__(self, model_name: str = 'u2net'):
        """
        Initialize background remover

        Args:
            model_name: Model to use. Options:
                - 'u2net': General purpose (default)
                - 'u2netp': Lightweight, faster
                - 'u2net_human_seg': Best for people/clothing
                - 'u2net_cloth_seg': Specifically for clothing
                - 'silueta': High quality
                - 'isnet-general-use': Good balance
                - 'isnet-anime': For anime/illustrated content
        """
        self.model_name = model_name
        logger.info(f"Initialized BackgroundRemover with model: {model_name}")

    def remove_background(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        alpha_matting: bool = True,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10
    ) -> Path:
        """
        Remove background from a single image

        Args:
            input_path: Path to input image
            output_path: Path to save output (PNG with transparency)
            alpha_matting: Enable alpha matting for better edges
            alpha_matting_foreground_threshold: Threshold for foreground
            alpha_matting_background_threshold: Threshold for background
            alpha_matting_erode_size: Erosion size for alpha matting

        Returns:
            Path to output image
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        logger.info(f"Processing: {input_path.name}")

        # Read image
        with open(input_path, 'rb') as f:
            input_data = f.read()

        # Remove background
        output_data = remove(
            input_data,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
            post_process_mask=True,  # Smooth the mask
            bgcolor=None  # Transparent background
        )

        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(output_data)

        logger.info(f"✅ Saved: {output_path}")
        return output_path

    def remove_background_pil(
        self,
        image: Image.Image,
        alpha_matting: bool = True
    ) -> Image.Image:
        """
        Remove background from PIL Image object

        Args:
            image: PIL Image object
            alpha_matting: Enable alpha matting

        Returns:
            PIL Image with transparent background
        """
        # Convert PIL image to bytes
        from io import BytesIO
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Remove background
        output_data = remove(
            img_byte_arr,
            alpha_matting=alpha_matting,
            post_process_mask=True
        )

        # Convert back to PIL
        output_image = Image.open(BytesIO(output_data))
        return output_image

    def batch_remove_backgrounds(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: List[str] = ['.jpg', '.jpeg', '.png'],
        alpha_matting: bool = True
    ):
        """
        Remove backgrounds from all images in a directory

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output images
            extensions: Image file extensions to process
            alpha_matting: Enable alpha matting
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

        logger.info(f"Found {len(image_files)} images to process")

        # Process each image
        success_count = 0
        for img_path in image_files:
            try:
                output_path = output_dir / f"{img_path.stem}_nobg.png"
                self.remove_background(
                    img_path,
                    output_path,
                    alpha_matting=alpha_matting
                )
                success_count += 1
            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {e}")

        logger.info(f"✅ Successfully processed {success_count}/{len(image_files)} images")

    def create_white_background(
        self,
        transparent_image_path: Union[str, Path],
        output_path: Union[str, Path],
        background_color: tuple = (255, 255, 255)
    ):
        """
        Add a solid color background to a transparent image

        Args:
            transparent_image_path: Path to image with transparent background
            output_path: Path to save output
            background_color: RGB tuple for background color
        """
        img = Image.open(transparent_image_path).convert('RGBA')

        # Create background
        background = Image.new('RGBA', img.size, background_color + (255,))

        # Composite
        result = Image.alpha_composite(background, img)
        result = result.convert('RGB')

        result.save(output_path)
        logger.info(f"✅ Created image with {background_color} background: {output_path}")


def compare_models(image_path: str, output_dir: str = "model_comparison"):
    """
    Compare different background removal models on the same image

    Args:
        image_path: Path to test image
        output_dir: Directory to save comparison results
    """
    models = [
        'u2net',
        'u2netp', 
        'u2net_human_seg',
        'u2net_cloth_seg',
        'silueta',
        'isnet-general-use'
    ]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(image_path).stem

    print(f"\n{'='*80}")
    print(f"COMPARING MODELS ON: {image_path}")
    print('='*80)

    for model in models:
        try:
            print(f"\nTesting model: {model}")
            remover = BackgroundRemover(model_name=model)
            output_path = output_dir / f"{image_name}_{model}.png"
            remover.remove_background(image_path, output_path)
        except Exception as e:
            print(f"  ❌ Error with {model}: {e}")

    print(f"\n✅ Comparison complete! Check {output_dir}")


if __name__ == "__main__":
    # Example usage

    # Method 1: Remove background from single image
    remover = BackgroundRemover(model_name='u2net_cloth_seg')
    remover.remove_background(
        'input.jpg',
        'output_nobg.png',
        alpha_matting=True
    )

    # Method 2: Batch process entire directory
    remover.batch_remove_backgrounds(
        input_dir='images',
        output_dir='images_nobg',
        alpha_matting=True
    )

    # Method 3: Compare different models
    compare_models('test_image.jpg', output_dir='comparison')

    # Method 4: Add white background to transparent image
    remover.create_white_background(
        'output_nobg.png',
        'output_white_bg.png',
        background_color=(255, 255, 255)
    )
