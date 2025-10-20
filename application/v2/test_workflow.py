# test_workflow.py
# Quick test with dummy data

import json
import numpy as np
import os

print("="*80)
print("TESTING WORKFLOW WITH DUMMY DATA")
print("="*80)

# Create dummy data directory
os.makedirs("data", exist_ok=True)

# Create 10 dummy items
print("\n📝 Creating dummy wardrobe data...")

dummy_items = [
    {"category": "kurti", "color_primary": "red", "occasions": ["wedding"], "weather": ["summer"], "style": "traditional"},
    {"category": "palazzo", "color_primary": "golden", "occasions": ["wedding"], "weather": ["summer"], "style": "traditional"},
    {"category": "dupatta", "color_primary": "red", "occasions": ["wedding"], "weather": ["all_season"], "style": "traditional"},
    {"category": "kurti", "color_primary": "blue", "occasions": ["party"], "weather": ["winter"], "style": "contemporary"},
    {"category": "palazzo", "color_primary": "black", "occasions": ["party"], "weather": ["winter"], "style": "contemporary"},
    {"category": "saree", "color_primary": "green", "occasions": ["festival"], "weather": ["all_season"], "style": "traditional"},
    {"category": "jeans", "color_primary": "blue", "occasions": ["casual"], "weather": ["all_season"], "style": "western"},
    {"category": "shirt", "color_primary": "white", "occasions": ["casual", "office"], "weather": ["all_season"], "style": "western"},
    {"category": "lehenga", "color_primary": "pink", "occasions": ["wedding"], "weather": ["winter"], "style": "traditional"},
    {"category": "jacket", "color_primary": "brown", "occasions": ["casual"], "weather": ["winter"], "style": "fusion"},
]

# Save as JSONL
with open("data/wardrobe_with_embeddings.jsonl", 'w') as f:
    for idx, item in enumerate(dummy_items):
        entry = {
            "filename": f"item_{idx}.jpg",
            "image_path": f"dummy_path_{idx}.jpg",
            "embedding_index": idx,
            "classification": item
        }
        f.write(json.dumps(entry) + '\n')

# Create dummy embeddings (random vectors for testing)
dummy_embeddings = np.random.randn(10, 512).astype(np.float32)
# Normalize
dummy_embeddings = dummy_embeddings / np.linalg.norm(dummy_embeddings, axis=1, keepdims=True)
np.save("data/wardrobe_embeddings.npy", dummy_embeddings)

print("✅ Created 10 dummy items with embeddings")

# Test outfit generator
print("\n🧪 Testing outfit generator...")

from outfit_generator import OutfitGenerator

generator = OutfitGenerator(
    metadata_jsonl="data/wardrobe_with_embeddings.jsonl",
    embeddings_npy="data/wardrobe_embeddings.npy"
)

outfits = generator.recommend_outfits(
    occasion='wedding',
    weather='summer',
    style='traditional',
    num_outfits=3
)

print(f"\n✅ Generated {len(outfits)} outfits!")

for i, outfit in enumerate(outfits, 1):
    print(f"\nOutfit {i}:")
    print(f"  Coherence: {outfit['visual_coherence']:.2%}")
    print(f"  Structure: {outfit['structure']}")
    for item in outfit['items']:
        c = item['classification']
        print(f"    - {c['category']}: {c['color_primary']}")

print("\n" + "="*80)
print("✅ WORKFLOW TEST COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Replace dummy data with your actual wardrobe_database.jsonl")
print("2. Run: python 1_generate_embeddings.py")
print("3. Run: python 2_outfit_generator.py")
print("4. Run: streamlit run 3_streamlit_app.py")
print("="*80)
