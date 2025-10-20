
# outfit_visualizer.py - Create stacked outfit images
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

class OutfitVisualizer:
    """Create visual representations of complete outfits"""

    def __init__(self, output_dir: str = "outfit_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def stitch_outfit_vertical(
        self, 
        outfit: Dict, 
        target_width: int = 400,
        add_border: bool = True,
        border_color: tuple = (255, 255, 255),
        border_width: int = 10
    ) -> Image.Image:
        """
        Stitch outfit items vertically to create a complete outfit view

        Args:
            outfit: Outfit dictionary with items
            target_width: Width of output image
            add_border: Add white border between items
            border_color: Color of border (RGB tuple)
            border_width: Width of border in pixels

        Returns:
            PIL Image of stitched outfit
        """
        items = outfit['items']
        images = []

        # Load and resize all images
        for item in items:
            img_path = item.get('image_path_nobg')
            if Path(img_path).exists():
                img = Image.open(img_path)
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Resize to target width, maintain aspect ratio
                aspect_ratio = img.height / img.width
                new_height = int(target_width * aspect_ratio)
                img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
                images.append(img)

        if not images:
            # Create placeholder if no images found
            placeholder = Image.new('RGB', (target_width, 600), (200, 200, 200))
            return placeholder

        # Calculate total height
        total_height = sum(img.height for img in images)
        if add_border and len(images) > 1:
            total_height += border_width * (len(images) - 1)

        # Create composite image
        composite = Image.new('RGB', (target_width, total_height), border_color)

        # Paste images vertically
        y_offset = 0
        for img in images:
            composite.paste(img, (0, y_offset))
            y_offset += img.height
            if add_border:
                y_offset += border_width

        return composite

    def create_outfit_with_overlay(
        self,
        outfit: Dict,
        target_width: int = 400,
        add_score_overlay: bool = True,
        add_labels: bool = True
    ) -> Image.Image:
        """
        Create outfit image with score overlay and item labels

        Args:
            outfit: Outfit dictionary
            target_width: Width of output image
            add_score_overlay: Add score badge on top
            add_labels: Add item category labels

        Returns:
            PIL Image with overlays
        """
        # First, stitch the outfit
        composite = self.stitch_outfit_vertical(outfit, target_width)

        # Create a drawing context
        draw = ImageDraw.Draw(composite)

        # Add score overlay (top-right corner)
        if add_score_overlay:
            score = outfit['score']
            score_text = f"{score.overall:.0%}"

            # Score badge dimensions
            badge_width = 80
            badge_height = 80
            badge_x = composite.width - badge_width - 10
            badge_y = 10

            # Draw badge background (semi-transparent circle)
            # Create overlay for transparency
            overlay = Image.new('RGBA', composite.size, (255, 255, 255, 0))
            overlay_draw = ImageDraw.Draw(overlay)

            # Draw circle
            circle_bbox = [
                badge_x,
                badge_y,
                badge_x + badge_width,
                badge_y + badge_height
            ]

            # Score color based on value
            if score.overall >= 0.8:
                badge_color = (34, 197, 94, 220)  # Green
            elif score.overall >= 0.6:
                badge_color = (59, 130, 246, 220)  # Blue
            else:
                badge_color = (251, 146, 60, 220)  # Orange

            overlay_draw.ellipse(circle_bbox, fill=badge_color)

            # Composite overlay
            composite = Image.alpha_composite(
                composite.convert('RGBA'), 
                overlay
            ).convert('RGB')

            # Re-create draw context for text
            draw = ImageDraw.Draw(composite)

            # Add score text (try to use a font, fallback to default)
            try:
                font = ImageFont.truetype("arial.ttf", 24)
                font_small = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
                font_small = ImageFont.load_default()

            # Draw text
            text_bbox = draw.textbbox((0, 0), score_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_x = badge_x + (badge_width - text_width) // 2
            text_y = badge_y + (badge_height - text_height) // 2 - 5

            draw.text((text_x, text_y), score_text, fill=(255, 255, 255), font=font)
            draw.text(
                (badge_x + 15, badge_y + badge_height - 25), 
                "MATCH", 
                fill=(255, 255, 255), 
                font=font_small
            )

        return composite

    def save_outfit_image(
        self,
        outfit: Dict,
        outfit_id: int,
        add_score: bool = True
    ) -> Path:
        """
        Save outfit as image file

        Args:
            outfit: Outfit dictionary
            outfit_id: Unique ID for filename
            add_score: Add score overlay

        Returns:
            Path to saved image
        """
        if add_score:
            img = self.create_outfit_with_overlay(outfit)
        else:
            img = self.stitch_outfit_vertical(outfit)

        # Generate filename
        filename = f"outfit_{outfit_id:03d}.png"
        filepath = self.output_dir / filename

        # Save
        img.save(filepath, 'PNG')
        print(f"✅ Saved outfit image: {filepath}")

        return filepath

    def create_outfit_grid(
        self,
        outfits: List[Dict],
        cols: int = 3,
        outfit_width: int = 300,
        spacing: int = 20
    ) -> Image.Image:
        """
        Create a grid of outfit images

        Args:
            outfits: List of outfits
            cols: Number of columns
            outfit_width: Width of each outfit
            spacing: Space between outfits

        Returns:
            PIL Image grid
        """
        if not outfits:
            return Image.new('RGB', (100, 100), (200, 200, 200))

        # Create individual outfit images
        outfit_images = []
        max_height = 0

        for outfit in outfits:
            img = self.create_outfit_with_overlay(outfit, target_width=outfit_width)
            outfit_images.append(img)
            max_height = max(max_height, img.height)

        # Calculate grid dimensions
        rows = (len(outfits) + cols - 1) // cols
        grid_width = cols * outfit_width + (cols - 1) * spacing
        grid_height = rows * max_height + (rows - 1) * spacing

        # Create grid image
        grid = Image.new('RGB', (grid_width, grid_height), (245, 245, 245))

        # Paste outfit images
        for idx, img in enumerate(outfit_images):
            row = idx // cols
            col = idx % cols

            x = col * (outfit_width + spacing)
            y = row * (max_height + spacing)

            grid.paste(img, (x, y))

        return grid
