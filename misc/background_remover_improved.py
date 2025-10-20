
# background_remover_improved.py - High quality background removal
from rembg import remove
from PIL import Image, ImageFilter
import numpy as np
from pathlib import Path
from typing import Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HighQualityBackgroundRemover:
    """
    High-quality background removal with anti-aliasing and edge refinement
    """

    def __init__(self, model_name: str = 'u2net_cloth_seg'):
        self.model_name = model_name
        logger.info(f"Initialized with model: {model_name}")

    def remove_background_hq(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        alpha_matting: bool = True,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10,
        post_process_mask: bool = True,
        smooth_edges: bool = True,
        edge_smooth_radius: int = 2
    ) -> Path:
        """
        Remove background with high quality settings

        Args:
            input_path: Input image path
            output_path: Output PNG path (must be .png)
            alpha_matting: Enable alpha matting for better edges
            alpha_matting_foreground_threshold: Threshold for foreground (default 240)
            alpha_matting_background_threshold: Threshold for background (default 10)
            alpha_matting_erode_size: Erosion size (default 10)
            post_process_mask: Apply post-processing to mask
            smooth_edges: Apply edge smoothing
            edge_smooth_radius: Radius for edge smoothing (1-3 recommended)

        Returns:
            Path to output PNG
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Force PNG output
        if output_path.suffix.lower() != '.png':
            output_path = output_path.with_suffix('.png')
            logger.warning(f"Changed output format to PNG: {output_path}")

        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        logger.info(f"Processing: {input_path.name}")

        # Read image
        input_image = Image.open(input_path)

        # Convert to RGB if necessary
        if input_image.mode == 'RGBA':
            # Create white background
            background = Image.new('RGB', input_image.size, (255, 255, 255))
            background.paste(input_image, mask=input_image.split()[3])
            input_image = background
        elif input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')

        # Save to bytes for rembg
        from io import BytesIO
        img_byte_arr = BytesIO()
        input_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Remove background
        output_data = remove(
            img_byte_arr,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
            post_process_mask=post_process_mask,
            bgcolor=None
        )

        # Load result
        output_image = Image.open(BytesIO(output_data))

        # Optional edge smoothing for better quality
        if smooth_edges and edge_smooth_radius > 0:
            output_image = self._smooth_edges(output_image, edge_smooth_radius)

        # Save as PNG with maximum quality
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_image.save(
            output_path,
            'PNG',
            optimize=False,  # Don't optimize to preserve quality
            compress_level=1  # Minimal compression for best quality
        )

        logger.info(f"✅ Saved high-quality PNG: {output_path}")
        return output_path

    def _smooth_edges(self, image: Image.Image, radius: int = 2) -> Image.Image:
        """
        Smooth edges of transparent image for better anti-aliasing

        Args:
            image: PIL Image with alpha channel
            radius: Smoothing radius (1-3 recommended)

        Returns:
            Image with smoothed edges
        """
        if image.mode != 'RGBA':
            return image

        # Extract alpha channel
        r, g, b, a = image.split()

        # Apply slight blur to alpha channel only
        a_smoothed = a.filter(ImageFilter.GaussianBlur(radius=radius * 0.5))

        # Recombine
        smoothed = Image.merge('RGBA', (r, g, b, a_smoothed))

        return smoothed

    def upscale_and_remove_bg(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        upscale_factor: float = 2.0,
        **kwargs
    ) -> Path:
        """
        Upscale image first, then remove background for better quality

        Args:
            input_path: Input image path
            output_path: Output PNG path
            upscale_factor: Factor to upscale before processing
            **kwargs: Additional arguments for remove_background_hq

        Returns:
            Path to output PNG
        """
        input_path = Path(input_path)

        # Load and upscale
        img = Image.open(input_path)
        original_size = img.size

        new_size = (
            int(img.width * upscale_factor),
            int(img.height * upscale_factor)
        )

        logger.info(f"Upscaling from {original_size} to {new_size}")

        # Use high-quality resampling
        img_upscaled = img.resize(new_size, Image.Resampling.LANCZOS)

        # Save temporary upscaled version
        temp_path = input_path.parent / f"temp_upscaled_{input_path.name}"
        img_upscaled.save(temp_path, quality=100)

        # Remove background from upscaled image
        result = self.remove_background_hq(temp_path, output_path, **kwargs)

        # Clean up temp file
        temp_path.unlink()

        return result

    def enhance_white_clothing(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> Path:
        """
        Special processing for white/light colored clothing
        Increases contrast before background removal

        Args:
            input_path: Input image path
            output_path: Output PNG path

        Returns:
            Path to output PNG
        """
        from PIL import ImageEnhance

        input_path = Path(input_path)

        # Load image
        img = Image.open(input_path)

        # Slightly increase contrast to help with edge detection
        enhancer = ImageEnhance.Contrast(img)
        img_enhanced = enhancer.enhance(1.1)  # 10% more contrast

        # Slightly increase sharpness
        enhancer = ImageEnhance.Sharpness(img_enhanced)
        img_enhanced = enhancer.enhance(1.2)  # 20% more sharpness

        # Save temporary enhanced version
        temp_path = input_path.parent / f"temp_enhanced_{input_path.name}"
        img_enhanced.save(temp_path, quality=100)

        # Remove background
        result = self.remove_background_hq(
            temp_path,
            output_path,
            alpha_matting=True,
            alpha_matting_foreground_threshold=230,  # Lower for white clothes
            smooth_edges=True,
            edge_smooth_radius=2
        )

        # Clean up
        temp_path.unlink()

        return result


# Convenience functions
def remove_bg_high_quality(input_path: str, output_path: str, model: str = 'u2net_cloth_seg'):
    """Quick high-quality background removal"""
    remover = HighQualityBackgroundRemover(model_name=model)
    return remover.remove_background_hq(input_path, output_path)


def remove_bg_white_clothing(input_path: str, output_path: str):
    """Special processing for white/cream colored clothing"""
    remover = HighQualityBackgroundRemover(model_name='u2net_cloth_seg')
    return remover.enhance_white_clothing(input_path, output_path)


if __name__ == "__main__":
    # Example usage
    remover = HighQualityBackgroundRemover(model_name='u2net_cloth_seg')

    # Method 1: Standard high quality
    remover.remove_background_hq(
        'kurti.jpg',
        'kurti_nobg.png',  # Always PNG!
        alpha_matting=True,
        smooth_edges=True
    )

    # Method 2: For white/light clothing
    remover.enhance_white_clothing(
        'white_kurti.jpg',
        'white_kurti_nobg.png'
    )

    # Method 3: Upscale first for better quality
    remover.upscale_and_remove_bg(
        'small_image.jpg',
        'large_nobg.png',
        upscale_factor=2.0
    )
