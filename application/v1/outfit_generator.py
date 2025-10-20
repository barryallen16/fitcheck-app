# 2_outfit_generator.py
# Core recommendation engine

import numpy as np
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
import os

print("="*80)
print("STEP 2: OUTFIT RECOMMENDATION ENGINE")
print("="*80)


class OutfitGenerator:
    """
    Generate complete outfit recommendations from wardrobe items
    """
    
    def __init__(self, metadata_jsonl, embeddings_npy):
        """
        Initialize with wardrobe data
        
        Args:
            metadata_jsonl: Path to wardrobe metadata with embeddings
            embeddings_npy: Path to CLIP embeddings array
        """
        print("\n📂 Loading wardrobe data...")
        
        # Load metadata
        self.wardrobe = []
        with open(metadata_jsonl, 'r') as f:
            for line in f:
                self.wardrobe.append(json.loads(line))
        
        # Load embeddings
        self.embeddings = np.load(embeddings_npy)
        
        print(f"✅ Loaded {len(self.wardrobe)} items")
        print(f"✅ Embeddings shape: {self.embeddings.shape}")
        
        # Organize by category type
        self._organize_by_category()
    
    
    def _organize_by_category(self):
        """Organize items by category type for efficient filtering"""
        self.items_by_type = {
            'tops': [],
            'bottoms': [],
            'one_piece': [],
            'layers': [],
            'western_tops': [],
            'western_bottoms': []
        }
        
        for idx, item in enumerate(self.wardrobe):
            # Get category type from classification
            if 'classification' in item:
                category = item['classification'].get('category', '')
            else:
                category = item.get('category', '')
            
            # Map to category type
            category_type = self._get_category_type(category)
            
            if category_type in self.items_by_type:
                item_with_idx = item.copy()
                item_with_idx['wardrobe_index'] = idx
                self.items_by_type[category_type].append(item_with_idx)
    
    
    def _get_category_type(self, category):
        """Map category to category type"""
        category = category.lower()
        
        if category in ['kurti', 'kurta', 'anarkali_top', 'blouse', 'choli', 'women_kurta', 'women_anarkali_kurta']:
            return 'tops'
        elif category in ['palazzo', 'churidar', 'salwar', 'leggings_salwar', 'dhoti_pants']:
            return 'bottoms'
        elif category in ['saree', 'lehenga', 'gown', 'lehenga_set']:
            return 'one_piece'
        elif category in ['dupatta', 'shawl', 'jacket', 'cape']:
            return 'layers'
        elif category in ['shirt', 't_shirt']:
            return 'western_tops'
        elif category in ['jeans', 'trousers', 'pants']:
            return 'western_bottoms'
        else:
            return 'other'
    
    def filter_by_preferences(self, occasion=None, weather=None, style=None):
        """
        Filter wardrobe by user preferences
        
        Returns filtered items organized by category type
        """
        filtered = {key: [] for key in self.items_by_type.keys()}
        
        for cat_type, items in self.items_by_type.items():
            for item in items:
                classification = item.get('classification', {})
                
                # Check occasion
                if occasion and occasion != 'any':
                    occasions = classification.get('occasions', [])
                    # Make sure occasions is a list
                    if isinstance(occasions, str):
                        occasions = [occasions]
                    
                    if not occasions:
                        continue
                        
                    # Check if any occasion matches
                    occasion_match = any(
                        occasion.lower() in str(occ).lower() 
                        for occ in occasions
                    )
                    if not occasion_match:
                        continue
                
                # Check weather
                if weather and weather != 'any':
                    weathers = classification.get('weather', [])
                    # Make sure weathers is a list
                    if isinstance(weathers, str):
                        weathers = [weathers]
                    
                    if not weathers:
                        continue
                    
                    # Check if any weather matches
                    weather_match = any(
                        weather.lower() in str(w).lower() 
                        for w in weathers
                    )
                    if not weather_match:
                        continue
                
                # Check style
                if style and style != 'any':
                    item_style = classification.get('style', '')
                    if not item_style:
                        continue
                    
                    # More flexible style matching
                    style_match = style.lower() in str(item_style).lower()
                    if not style_match:
                        continue
                
                filtered[cat_type].append(item)
        
        return filtered    
    
    def generate_outfit_combinations(self, filtered_items, max_combinations=50):
        """
        Generate possible outfit combinations from filtered items
        """
        combinations = []
        
        # Traditional ethnic: Top + Bottom + Layer
        tops = filtered_items['tops'][:10]
        bottoms = filtered_items['bottoms'][:10]
        layers = filtered_items['layers'][:5]
        
        for top in tops:
            for bottom in bottoms:
                # Without layer
                combinations.append({
                    'items': [top, bottom],
                    'structure': 'top + bottom',
                    'type': 'ethnic_separates'
                })
                
                # With layer
                for layer in layers:
                    combinations.append({
                        'items': [top, bottom, layer],
                        'structure': 'top + bottom + layer',
                        'type': 'ethnic_layered'
                    })
        
        # One-piece outfits
        one_pieces = filtered_items['one_piece'][:10]
        
        for piece in one_pieces:
            # Alone
            combinations.append({
                'items': [piece],
                'structure': 'one_piece',
                'type': 'traditional'
            })
            
            # With layer
            for layer in layers[:3]:
                combinations.append({
                    'items': [piece, layer],
                    'structure': 'one_piece + layer',
                    'type': 'traditional_layered'
                })
        
        # Western outfits
        western_tops = filtered_items['western_tops'][:10]
        western_bottoms = filtered_items['western_bottoms'][:10]
        
        for wtop in western_tops:
            for wbottom in western_bottoms:
                combinations.append({
                    'items': [wtop, wbottom],
                    'structure': 'western_top + western_bottom',
                    'type': 'western'
                })
        
        return combinations[:max_combinations]
    
    
    def compute_visual_coherence(self, outfit_items):
        """
        Compute how visually harmonious the outfit is using CLIP embeddings
        """
        if len(outfit_items) == 1:
            return 1.0
        
        # Get embedding indices
        indices = [item['embedding_index'] for item in outfit_items]
        outfit_embeddings = self.embeddings[indices]
        
        # Compute pairwise similarities
        similarities = cosine_similarity(outfit_embeddings)
        
        # Average similarity (excluding diagonal)
        n = len(outfit_items)
        total_sim = (similarities.sum() - n) / (n * (n - 1))
        
        return float(total_sim)
    
    
    def rank_outfits(self, combinations):
        """
        Rank outfits by visual coherence
        """
        for combo in combinations:
            combo['visual_coherence'] = self.compute_visual_coherence(combo['items'])
        
        # Sort by coherence
        combinations.sort(key=lambda x: x['visual_coherence'], reverse=True)
        
        return combinations
    
    
    def recommend_outfits(self, occasion='any', weather='any', style='any', num_outfits=5):
        """
        Main recommendation function
        
        Returns:
            List of complete outfit recommendations
        """
        print(f"\n🔍 Filtering by: occasion={occasion}, weather={weather}, style={style}")
        
        # Filter items
        filtered = self.filter_by_preferences(occasion, weather, style)
        
        total_filtered = sum(len(items) for items in filtered.values())
        print(f"✅ Found {total_filtered} matching items")
        
        if total_filtered < 2:
            print("❌ Not enough items for outfits")
            return []
        
        # Generate combinations
        print(f"🎨 Generating outfit combinations...")
        combinations = self.generate_outfit_combinations(filtered)
        print(f"✅ Generated {len(combinations)} combinations")
        
        if not combinations:
            return []
        
        # Rank by visual coherence
        print(f"📊 Ranking outfits...")
        ranked = self.rank_outfits(combinations)
        
        # Return top N
        top_outfits = ranked[:num_outfits]
        
        print(f"✅ Returning top {len(top_outfits)} outfits")
        
        return top_outfits


# ============================================================================
# Test the Engine
# ============================================================================

if __name__ == "__main__":
    
    # Initialize generator
    generator = OutfitGenerator(
        metadata_jsonl="data/men/men_wardrobe_with_embeddings.jsonl",
        embeddings_npy="data/men/men_wardrobe_embeddings.npy"
    )
    
    # Test recommendations
    print("\n" + "="*80)
    print("TESTING OUTFIT RECOMMENDATIONS")
    print("="*80)
    
    # Test 1: Use 'any' to see all combinations
    print("\n📊 TEST 1: All items (no filters)")
    outfits = generator.recommend_outfits(
        occasion='any',
        weather='any',
        style='any',
        num_outfits=5
    )
    print(f"Found {len(outfits)} outfits\n")
    
    # Test 2: Casual contemporary (most of your items)
    print("📊 TEST 2: Casual contemporary style")
    outfits = generator.recommend_outfits(
        occasion='casual',
        weather='all_season',
        style='contemporary',
        num_outfits=5
    )
    print(f"Found {len(outfits)} outfits\n")
    
    # Test 3: Traditional formal
    print("📊 TEST 3: Traditional formal style")
    outfits = generator.recommend_outfits(
        occasion='formal',
        weather='any',
        style='traditional',
        num_outfits=5
    )
    
    # Display results
    print("\n" + "="*80)
    print("RECOMMENDED OUTFITS")
    print("="*80)
    
    for i, outfit in enumerate(outfits, 1):
        print(f"\nOUTFIT #{i}")
        print(f"Type: {outfit['type']}")
        print(f"Structure: {outfit['structure']}")
        print(f"Visual Coherence: {outfit['visual_coherence']:.2%}")
        print(f"Items:")
        
        for item in outfit['items']:
            classification = item.get('classification', {})
            print(f"  • {classification.get('category', 'Unknown')}: {classification.get('color_primary', 'Unknown')} {classification.get('specific_type', '')}")
            print(f"    Style: {classification.get('style')}, Occasions: {classification.get('occasions')}")
        
        print("-" * 40)
    
    print("\n🎉 Step 2 complete! Run 3_streamlit_app.py for web interface.")