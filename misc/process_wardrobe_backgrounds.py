
# process_wardrobe_backgrounds.py
# Batch process all wardrobe images to remove backgrounds

from background_remover_improved import HighQualityBackgroundRemover
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_wardrobe_images(
    wardrobe_jsonl: str,
    output_dir: str = "images_nobg",
    model_name: str = 'u2net_cloth_seg',
    update_database: bool = True
):
    """
    Process all images in wardrobe database to remove backgrounds

    Args:
        wardrobe_jsonl: Path to wardrobe database JSONL file
        output_dir: Directory to save processed images
        model_name: Background removal model to use
        update_database: Whether to update database with new paths
    """

    # Initialize background remover
    remover = HighQualityBackgroundRemover(model_name=model_name)


    # Load wardrobe database
    wardrobe_items = []
    with open(wardrobe_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                wardrobe_items.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(wardrobe_items)} items from database")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each item
    processed_count = 0
    updated_items = []

    for item in wardrobe_items:
        try:
            # Get original image path
            img_path = Path(item.get('image_path', ''))

            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                updated_items.append(item)
                continue

            # Create output path
            output_file = output_path / f"{img_path.stem}_nobg.png"

            # Detect light-colored clothing
            color = item.get('classification', {}).get('color_primary', '').lower()
            is_light = any(c in color for c in ['white', 'cream', 'ivory', 'beige', 'light'])

            logger.info(f"Processing: {img_path.name} (color: {color})")

            if is_light:
                logger.info(f"  → Using white clothing enhancement")
                remover.enhance_white_clothing(img_path, output_file)
            else:
                logger.info(f"  → Using high-quality removal")
                remover.remove_background_hq(
                    input_path=img_path,
                    output_path=output_file,
                    alpha_matting=True,
                    smooth_edges=True,
                    edge_smooth_radius=2
                )


            # Update item with new path
            item_copy = item.copy()
            item_copy['image_path_nobg'] = str(output_file)
            item_copy['has_transparent_bg'] = True
            item_copy['bg_removal_quality'] = 'high'

            updated_items.append(item_copy)

            processed_count += 1

        except Exception as e:
            logger.error(f"Error processing {item.get('filename')}: {e}")
            updated_items.append(item)

    logger.info(f"✅ Processed {processed_count}/{len(wardrobe_items)} images")

    # Optionally update database
    if update_database:
        output_jsonl = wardrobe_jsonl.replace('.jsonl', '_with_nobg.jsonl')
        with open(output_jsonl, 'w') as f:
            for item in updated_items:
                f.write(json.dumps(item) + '\n')
        logger.info(f"✅ Updated database saved to: {output_jsonl}")

    return updated_items


def create_comparison_images(
    test_image_path: str,
    models: list = None
):
    """
    Compare different models on a test image

    Args:
        test_image_path: Path to test image
        models: List of model names to compare
    """
    if models is None:
        models = ['u2net', 'u2net_cloth_seg', 'u2net_human_seg', 'isnet-general-use']

    output_dir = Path("model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_path = Path(test_image_path)

    print(f"\n{'='*80}")
    print(f"COMPARING MODELS ON: {test_path.name}")
    print('='*80)

    for model in models:
        try:
            print(f"\n📊 Testing: {model}")
            remover = HighQualityBackgroundRemover(model_name=model)

            output_file = output_dir / f"{test_path.stem}_{model}.png"
            remover.remove_background_hq(test_path, output_file,
                              alpha_matting=True, smooth_edges=True)

            print(f"   ✅ Saved: {output_file}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print(f"\n✅ Comparison complete! Check: {output_dir}")
    print(f"\nOpen the images side-by-side to choose the best model.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python process_wardrobe_backgrounds.py <wardrobe.jsonl>")
        print("  python process_wardrobe_backgrounds.py compare <test_image.jpg>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "compare":
        # Compare models on test image
        if len(sys.argv) < 3:
            print("Please provide test image path")
            sys.exit(1)
        test_image = sys.argv[2]
        create_comparison_images(test_image)

    else:
        # Process entire wardrobe
        wardrobe_file = command

        # Optional: specify model
        model = sys.argv[2] if len(sys.argv) > 2 else 'u2net_cloth_seg'

        print(f"\n{'='*80}")
        print("PROCESSING WARDROBE IMAGES")
        print('='*80)
        print(f"Input: {wardrobe_file}")
        print(f"Model: {model}")
        print('='*80)

        process_wardrobe_images(
            wardrobe_jsonl=wardrobe_file,
            output_dir="images_nobg",
            model_name=model,
            update_database=True
        )

        print(f"\n✅ COMPLETE!")
        print(f"\nProcessed images saved to: images_nobg/")
        print(f"Updated database: {wardrobe_file.replace('.jsonl', '_with_nobg.jsonl')}")
