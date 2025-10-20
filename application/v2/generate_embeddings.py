# 1_generate_embeddings.py - Improved Version
# Generates CLIP embeddings with resume capability, validation, and error handling

import torch
import open_clip
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm
import time
from pathlib import Path
import hashlib
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate and manage fashion image embeddings"""
    
    def __init__(self, model_name: str = 'hf-hub:Marqo/marqo-fashionSigLIP'):
        """
        Initialize embedding generator
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.preprocess = None
        self._load_model()
    
    def _load_model(self):
        """Load CLIP model with error handling"""
        logger.info(f"Loading model: {self.model_name}")
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ Model loaded on {self.device.upper()}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def validate_image(self, image_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate image file
        
        Returns:
            (is_valid, error_message)
        """
        # Check file exists
        if not os.path.exists(image_path):
            return False, "File not found"
        
        # Check file size
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            return False, "Empty file"
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            return False, "File too large"
        
        # Check if valid image
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True, None
        except Exception as e:
            return False, f"Invalid image: {str(e)}"
    
    def generate_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Generate embedding for single image with validation
        
        Returns:
            768-dim embedding or None if failed
        """
        # Validate image
        is_valid, error = self.validate_image(image_path)
        if not is_valid:
            logger.warning(f"Skipping {image_path}: {error}")
            return None
        
        try:
            # Load and preprocess
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input, normalize=True)
            
            embedding = image_features[0].cpu().numpy()
            
            # Validate embedding
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                logger.warning(f"Invalid embedding for {image_path}")
                return None
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None
    
    def compute_file_hash(self, filepath: str) -> str:
        """Compute MD5 hash of file for change detection"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


def load_existing_data(
    output_jsonl: str, 
    output_embeddings: str
) -> Tuple[Dict[str, int], List[np.ndarray], List[Dict]]:
    """
    Load existing embeddings and metadata for resume capability
    
    Returns:
        (processed_paths_map, embeddings_list, metadata_list)
    """
    processed_paths = {}
    embeddings_list = []
    metadata_list = []
    
    if os.path.exists(output_jsonl) and os.path.exists(output_embeddings):
        logger.info("Loading existing data for resume...")
        
        try:
            # Load embeddings
            embeddings_array = np.load(output_embeddings)
            embeddings_list = list(embeddings_array)
            
            # Load metadata
            with open(output_jsonl, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        metadata_list.append(item)
                        processed_paths[item['image_path']] = item['embedding_index']
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"✅ Loaded {len(metadata_list)} existing items")
            
        except Exception as e:
            logger.warning(f"Could not load existing data: {e}")
            processed_paths = {}
            embeddings_list = []
            metadata_list = []
    
    return processed_paths, embeddings_list, metadata_list


def process_wardrobe(
    input_jsonl: str,
    output_embeddings: str,
    output_metadata: str,
    generator: EmbeddingGenerator,
    resume: bool = True,
    validate_existing: bool = False
):
    """
    Generate embeddings for wardrobe with resume capability
    
    Args:
        input_jsonl: Input classification data
        output_embeddings: Output .npy file
        output_metadata: Output JSONL with embedding indices
        generator: EmbeddingGenerator instance
        resume: Continue from previous run
        validate_existing: Re-validate existing embeddings
    """
    
    logger.info(f"Loading data from: {input_jsonl}")
    
    # Validate input file
    if not os.path.exists(input_jsonl):
        logger.error(f"Input file not found: {input_jsonl}")
        raise FileNotFoundError(f"Input file not found: {input_jsonl}")
    
    # Load input data
    input_data = []
    with open(input_jsonl, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                if 'image_path' in item:
                    input_data.append(item)
                else:
                    logger.warning(f"Line {line_num}: No image_path field")
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")
    
    logger.info(f"✅ Loaded {len(input_data)} items from input")
    
    # Load existing data if resuming
    processed_paths = {}
    embeddings_list = []
    metadata_list = []
    
    if resume:
        processed_paths, embeddings_list, metadata_list = load_existing_data(
            output_metadata, output_embeddings
        )
    
    # Filter items to process
    items_to_process = [
        item for item in input_data
        if item['image_path'] not in processed_paths
    ]
    
    logger.info(f"📊 Already processed: {len(processed_paths)}")
    logger.info(f"📊 New items to process: {len(items_to_process)}")
    
    if not items_to_process:
        logger.info("✅ All items already processed!")
        return
    
    # Process new items
    start_time = time.time()
    successful = 0
    failed = 0
    
    logger.info("🎨 Generating embeddings...")
    
    for item in tqdm(items_to_process, desc="Processing"):
        image_path = item['image_path']
        
        # Generate embedding
        embedding = generator.generate_embedding(image_path)
        
        if embedding is not None:
            # Add to lists
            embeddings_list.append(embedding)
            item['embedding_index'] = len(embeddings_list) - 1
            metadata_list.append(item)
            
            # Append to output file immediately (crash safety)
            with open(output_metadata, 'a') as f:
                f.write(json.dumps(item) + '\n')
            
            successful += 1
        else:
            failed += 1
        
        # Save checkpoint every 100 items
        if (successful + failed) % 100 == 0:
            embeddings_array = np.array(embeddings_list)
            np.save(output_embeddings, embeddings_array)
            logger.info(f"Checkpoint: {successful} success, {failed} failed")
    
    # Final save
    embeddings_array = np.array(embeddings_list)
    np.save(output_embeddings, embeddings_array)
    
    # Rewrite complete metadata file (sorted)
    with open(output_metadata, 'w') as f:
        for item in metadata_list:
            f.write(json.dumps(item) + '\n')
    
    elapsed = time.time() - start_time
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ COMPLETE!")
    logger.info(f"{'='*80}")
    logger.info(f"✅ Total items: {len(metadata_list)}")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"✅ Embeddings shape: {embeddings_array.shape}")
    logger.info(f"✅ Time: {elapsed/60:.1f} minutes")
    if successful > 0:
        logger.info(f"✅ Avg: {elapsed/successful:.2f} sec/image")
    logger.info(f"✅ Saved to: {output_embeddings}")
    logger.info(f"✅ Metadata: {output_metadata}")
    logger.info(f"{'='*80}\n")


def main():
    """Main execution"""
    
    print("="*80)
    print("FASHION EMBEDDING GENERATOR v2.0")
    print("="*80)
    
    # Create output directory
    os.makedirs("data", exist_ok=True)
    
    # Initialize generator
    generator = EmbeddingGenerator()
    
    # Configuration
    configs = [
        {
            'input': 'data/men_wardrobe_database.jsonl',
            'embeddings': 'data/men_wardrobe_embeddings.npy',
            'metadata': 'data/men_wardrobe_with_embeddings.jsonl'
        },
        # Add women's wardrobe if needed
        # {
        #     'input': 'data/women_wardrobe_database.jsonl',
        #     'embeddings': 'data/women_wardrobe_embeddings.npy',
        #     'metadata': 'data/women_wardrobe_with_embeddings.jsonl'
        # }
    ]
    
    # Process each configuration
    for config in configs:
        if os.path.exists(config['input']):
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing: {config['input']}")
            logger.info(f"{'='*80}")
            
            try:
                process_wardrobe(
                    input_jsonl=config['input'],
                    output_embeddings=config['embeddings'],
                    output_metadata=config['metadata'],
                    generator=generator,
                    resume=True
                )
            except Exception as e:
                logger.error(f"Failed to process {config['input']}: {e}")
        else:
            logger.warning(f"Skipping {config['input']} - file not found")
    
    print("\n🎉 All processing complete!")
    print("📝 Next step: Run 2_outfit_generator.py")


if __name__ == "__main__":
    main()