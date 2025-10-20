# 1_generate_embeddings.py
# Run this FIRST to generate CLIP embeddings from your classified data

import torch
import open_clip
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm
import time

print("="*80)
print("STEP 1: GENERATING MARQO-FASHIONSIGLIP EMBEDDINGS")
print("="*80)

# ============================================================================
# Load Marqo-FashionSigLIP Model
# ============================================================================

print("\n📦 Loading Marqo-FashionSigLIP model...")
try:
    model, _, preprocess = open_clip.create_model_and_transforms(
        'hf-hub:Marqo/marqo-fashionSigLIP'
    )
    
    # Move to GPU if available, otherwise CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully on {device.upper()}!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("💡 Try: pip install open-clip-torch")
    exit(1)


# ============================================================================
# Generate Embedding Function
# ============================================================================

def generate_embedding(image_path):
    """Generate 512-dim embedding for an image"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input, normalize=True)
        
        return image_features[0].cpu().numpy()
    except Exception as e:
        print(f"⚠️ Error processing {image_path}: {e}")
        return None


# ============================================================================
# Process Wardrobe Database
# ============================================================================

def process_wardrobe(
    input_jsonl="data/wardrobe_database.jsonl",
    output_embeddings="data/wardrobe_embeddings.npy",
    output_metadata="data/wardrobe_with_embeddings.jsonl"
):
    """
    Generate embeddings for all items in wardrobe database
    """
    
    print(f"\n📂 Loading data from: {input_jsonl}")
    
    # Check if file exists
    if not os.path.exists(input_jsonl):
        print(f"❌ File not found: {input_jsonl}")
        print("\n💡 Expected format (JSONL):")
        print('''
{
  "filename": "kurti_001.jpg",
  "image_path": "/path/to/image.jpg",
  "classification": {
    "category": "kurti",
    "color_primary": "maroon",
    ...
  }
}
        ''')
        exit(1)
    
    # Load data
    data_list = []
    with open(input_jsonl, 'r') as f:
        for line in f:
            try:
                data_list.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    print(f"✅ Loaded {len(data_list)} items")
    
    # Generate embeddings
    embeddings = []
    enriched_data = []
    
    print(f"\n🎨 Generating embeddings...")
    start_time = time.time()
    
    for idx, item in enumerate(tqdm(data_list, desc="Processing")):
        
        # Get image path
        if 'image_path' in item:
            image_path = item['image_path']
        else:
            print(f"⚠️ No image_path in item {idx}, skipping")
            continue
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}, skipping")
            continue
        
        # Generate embedding
        embedding = generate_embedding(image_path)
        
        if embedding is not None:
            embeddings.append(embedding)
            item['embedding_index'] = len(embeddings) - 1
            enriched_data.append(item)
    
    elapsed = time.time() - start_time
    
    # Save embeddings
    embeddings_array = np.array(embeddings)
    np.save(output_embeddings, embeddings_array)
    
    # Save metadata
    with open(output_metadata, 'w') as f:
        for item in enriched_data:
            f.write(json.dumps(item) + '\n')
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ COMPLETE!")
    print(f"{'='*80}")
    print(f"✅ Processed: {len(enriched_data)} items")
    print(f"✅ Embeddings shape: {embeddings_array.shape}")
    print(f"✅ Time: {elapsed/60:.1f} minutes")
    print(f"✅ Avg: {elapsed/len(enriched_data):.2f} sec/image")
    print(f"✅ Saved to: {output_embeddings}")
    print(f"✅ Metadata: {output_metadata}")
    print(f"{'='*80}\n")
    
    return embeddings_array, enriched_data


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    
    # Create data directory if not exists
    os.makedirs("data", exist_ok=True)
    
    # Process your wardrobe
    embeddings, metadata = process_wardrobe(
        input_jsonl="data/men_wardrobe_database.jsonl",  # Your classification results
        output_embeddings="data/men_wardrobe_embeddings.npy",
        output_metadata="data/men_wardrobe_with_embeddings.jsonl"
        #         input_jsonl="data/women_wardrobe_database.jsonl",  # Your classification results
        # output_embeddings="data/women_wardrobe_embeddings.npy",
        # output_metadata="data/women_wardrobe_with_embeddings.jsonl"
    )
    
    print("🎉 Step 1 complete! Run 2_outfit_generator.py next.")
