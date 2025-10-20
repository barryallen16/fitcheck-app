# 2_outfit_generator.py - Improved Version
# Advanced outfit recommendation with color theory, style rules, and smart matching

import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColorHarmony:
    """Color theory and harmony rules"""
    
    # Color families (simplified)
    COLOR_FAMILIES = {
        'neutral': ['white', 'black', 'grey', 'gray', 'beige', 'cream', 'ivory'],
        'warm': ['red', 'orange', 'yellow', 'maroon', 'mustard', 'gold', 'brown', 'rust'],
        'cool': ['blue', 'green', 'purple', 'navy', 'teal', 'turquoise', 'violet'],
        'earth': ['brown', 'olive', 'khaki', 'tan', 'camel', 'terracotta']
    }
    
    @classmethod
    def get_color_family(cls, color: str) -> str:
        """Get color family for a color"""
        color_lower = color.lower()
        for family, colors in cls.COLOR_FAMILIES.items():
            if any(c in color_lower for c in colors):
                return family
        return 'other'
    
    @classmethod
    def are_harmonious(cls, color1: str, color2: str) -> Tuple[bool, float]:
        """
        Check if two colors are harmonious
        
        Returns:
            (is_harmonious, harmony_score)
        """
        family1 = cls.get_color_family(color1)
        family2 = cls.get_color_family(color2)
        
        # Same color family - high harmony
        if family1 == family2:
            return True, 0.9
        
        # Neutral with anything - always works
        if 'neutral' in (family1, family2):
            return True, 0.85
        
        # Warm with earth tones
        if set([family1, family2]) == {'warm', 'earth'}:
            return True, 0.8
        
        # Cool colors together
        if set([family1, family2]) <= {'cool', 'neutral'}:
            return True, 0.75
        
        # Warm and cool - can work but lower score
        if set([family1, family2]) == {'warm', 'cool'}:
            return False, 0.4
        
        return True, 0.6


class StyleRules:
    """Fashion style compatibility rules"""
    
    STYLE_COMPATIBILITY = {
        'traditional': ['traditional', 'fusion', 'festive', 'casual'],  # Added 'casual'
        'contemporary': ['contemporary', 'fusion', 'casual', 'formal'],
        'fusion': ['traditional', 'contemporary', 'fusion', 'casual', 'western'],  # Added 'western'
        'casual': ['casual', 'contemporary', 'western', 'fusion', 'traditional'],  # Added 'traditional'
        'formal': ['formal', 'contemporary', 'traditional'],
        'festive': ['festive', 'traditional', 'fusion'],
        'western': ['western', 'casual', 'contemporary', 'traditional']  # Added 'traditional'
    }

    @classmethod
    def are_compatible(cls, style1: str, style2: str) -> Tuple[bool, float]:
        """
        Check if two styles are compatible
        
        Returns:
            (is_compatible, compatibility_score)
        """
        style1_lower = style1.lower()
        style2_lower = style2.lower()
        
        if style1_lower == style2_lower:
            return True, 1.0
        
        compatible_styles = cls.STYLE_COMPATIBILITY.get(style1_lower, [])
        
        if style2_lower in compatible_styles:
            # Direct match
            return True, 0.85
        
        # Check reverse
        compatible_styles_2 = cls.STYLE_COMPATIBILITY.get(style2_lower, [])
        if style1_lower in compatible_styles_2:
            return True, 0.85
        
        return False, 0.3


@dataclass
class OutfitScore:
    """Comprehensive outfit scoring"""
    visual_coherence: float  # CLIP embedding similarity
    color_harmony: float     # Color theory score
    style_compatibility: float  # Style matching score
    occasion_fit: float      # How well it fits the occasion
    overall: float           # Weighted average
    
    def to_dict(self) -> Dict:
        return {
            'visual_coherence': round(self.visual_coherence, 3),
            'color_harmony': round(self.color_harmony, 3),
            'style_compatibility': round(self.style_compatibility, 3),
            'occasion_fit': round(self.occasion_fit, 3),
            'overall': round(self.overall, 3)
        }


class OutfitGenerator:
    """Advanced outfit recommendation engine"""
    
    def __init__(self, metadata_jsonl: str, embeddings_npy: str):
        """Initialize with wardrobe data"""
        logger.info("Loading wardrobe data...")
        
        # Load metadata
        self.wardrobe = []
        with open(metadata_jsonl, 'r') as f:
            for line in f:
                try:
                    self.wardrobe.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        # Load embeddings
        self.embeddings = np.load(embeddings_npy)
        
        logger.info(f"✅ Loaded {len(self.wardrobe)} items")
        logger.info(f"✅ Embeddings shape: {self.embeddings.shape}")
        
        # Organize by category
        self._organize_wardrobe()
    
    def _organize_wardrobe(self):
        """Organize items by category type"""
        self.items_by_type = {
            'ethnic_tops': [],      # Kurtas, kurtis
            'ethnic_bottoms': [],   # Palazzo, churidar, salwar
            'ethnic_full': [],      # Saree, lehenga, sherwani
            'western_tops': [],     # Shirts, t-shirts
            'western_bottoms': [],  # Jeans, trousers
            'layers': [],           # Jackets, dupattas, shawls
            'other': []
        }
        
        category_mapping = {
            'women_kurta': 'ethnic_tops',
            'women_anarkali_kurta': 'ethnic_tops',
            'women_a_line_kurta': 'ethnic_tops',
            'men_kurta': 'ethnic_tops',
            'kurti': 'ethnic_tops',
            'kurta': 'ethnic_tops',
            'blouse': 'ethnic_tops',
            'choli': 'ethnic_tops',
            
            'palazzo': 'ethnic_bottoms',
            'leggings_salwar': 'ethnic_bottoms',
            'churidar': 'ethnic_bottoms',
            'salwar': 'ethnic_bottoms',
            'dhoti_pants': 'ethnic_bottoms',
            
            'saree': 'ethnic_full',
            'lehenga': 'ethnic_full',
            'sherwani': 'ethnic_full',
            'gown': 'ethnic_full',
            
            'shirt': 'western_tops',
            't_shirt': 'western_tops',
            
            'jeans': 'western_bottoms',
            'trousers': 'western_bottoms',
            'pants': 'western_bottoms',
            
            'nehru_jacket': 'layers',
            'jacket': 'layers',
            'dupatta': 'layers',
            'shawl': 'layers'
        }
        
        for idx, item in enumerate(self.wardrobe):
            classification = item.get('classification', {})
            category = classification.get('category', '').lower()
            
            item_type = category_mapping.get(category, 'other')
            
            item_with_idx = item.copy()
            item_with_idx['wardrobe_index'] = idx
            self.items_by_type[item_type].append(item_with_idx)
        
        # Log statistics
        for item_type, items in self.items_by_type.items():
            if items:
                logger.info(f"  {item_type}: {len(items)} items")
    
# Corrected code for outfit_generator.py

    def filter_by_preferences(
        self, 
        occasion: Optional[str] = None,
        weather: Optional[str] = None,
        style: Optional[str] = None,
        gender: Optional[str] = None,
        formality: Optional[str] = None
    ) -> Dict[str, List]:
        """Advanced filtering with multiple criteria using exact matching."""
        
        filtered = {key: [] for key in self.items_by_type.keys()}
        
        # Prepare lowercase filter values once
        occasion_filter = occasion.lower() if occasion and occasion != 'any' else None
        weather_filter = weather.lower() if weather and weather != 'any' else None
        style_filter = style.lower() if style and style != 'any' else None
        gender_filter = gender.lower() if gender and gender != 'any' else None
        formality_filter = formality.lower() if formality and formality != 'any' else None

        for item_type, items in self.items_by_type.items():
            for item in items:
                classification = item.get('classification', {})
                
                # Occasion filter (exact match in list)
                if occasion_filter:
                    occasions = classification.get('occasions', [])
                    if not isinstance(occasions, list):
                        occasions = [occasions]
                    if occasion_filter not in [str(o).lower() for o in occasions]:
                        continue
                
                # *** NEW WEATHER LOGIC ***
                if weather_filter:
                    item_weathers = classification.get('weather', [])
                    if not isinstance(item_weathers, list):
                        item_weathers = [item_weathers]
                    
                    # Normalize item weathers to lowercase
                    item_weathers_lower = [str(w).lower() for w in item_weathers]
                    
                    # An item matches if its list contains the specific season OR 'all_season'
                    if weather_filter not in item_weathers_lower and 'all_season' not in item_weathers_lower:
                        continue
                # *** END NEW WEATHER LOGIC ***
                
                # Style filter (exact match for string)
                if style_filter and style_filter != 'fusion':
                    item_style = classification.get('style', '')
                    if style_filter != str(item_style).lower():
                        continue
                
                # Gender filter (exact match for string)
                if gender_filter:
                    item_gender = classification.get('gender', '')
                    if str(item_gender).lower() != 'unisex' and gender_filter != str(item_gender).lower():
                        continue
                
                # Formality filter (exact match for string)
                if formality_filter:
                    item_formality = classification.get('formality', '')
                    if formality_filter != str(item_formality).lower():
                        continue
                
                filtered[item_type].append(item)
        
        return filtered
    
    def generate_combinations(self, filtered: Dict, max_per_type: int = 100) -> List[Dict]:
        """Generate smart outfit combinations"""
        combinations = []
        
        # Limit items per type for performance
        ethnic_tops = filtered['ethnic_tops'][:max_per_type]
        ethnic_bottoms = filtered['ethnic_bottoms'][:max_per_type]
        ethnic_full = filtered['ethnic_full'][:max_per_type]
        western_tops = filtered['western_tops'][:max_per_type]
        western_bottoms = filtered['western_bottoms'][:max_per_type]
        layers = filtered['layers'][:max_per_type]
        
        # Ethnic: Top + Bottom
        for top in ethnic_tops:
            for bottom in ethnic_bottoms:
                combinations.append({
                    'items': [top, bottom],
                    'type': 'ethnic_separates',
                    'structure': 'top + bottom'
                })
        
        # Ethnic: Top + Bottom + Layer
        for top in ethnic_tops[:10]:
            for bottom in ethnic_bottoms[:10]:
                for layer in layers[:5]:
                    combinations.append({
                        'items': [top, bottom, layer],
                        'type': 'ethnic_layered',
                        'structure': 'top + bottom + layer'
                    })
        
        # Ethnic full outfits
        for full in ethnic_full:
            combinations.append({
                'items': [full],
                'type': 'traditional_full',
                'structure': 'complete_outfit'
            })
            
            # Full + Layer
            for layer in layers[:3]:
                combinations.append({
                    'items': [full, layer],
                    'type': 'traditional_layered',
                    'structure': 'outfit + layer'
                })
        
        # Western: Top + Bottom
        for top in western_tops:
            for bottom in western_bottoms:
                combinations.append({
                    'items': [top, bottom],
                    'type': 'western',
                    'structure': 'shirt + pants'
                })
        # Fusion combinations: Mix ethnic and western
# Fusion: Ethnic Top + Western Bottom (Kurti + Jeans)
        for top in ethnic_tops:
            for bottom in western_bottoms:
                combinations.append({
                    'items': [top, bottom],
                    'type': 'fusion',
                    'structure': 'ethnic_top + western_bottom'
                })

        # Fusion: Western Top + Ethnic Bottom (Shirt + Palazzo)
        for top in western_tops:
            for bottom in ethnic_bottoms:
                combinations.append({
                    'items': [top, bottom],
                    'type': 'fusion',
                    'structure': 'western_top + ethnic_bottom'
                })

        # Fusion with layers (optional, for variety)
        for top in ethnic_tops[:10]:
            for bottom in western_bottoms[:10]:
                for layer in layers[:5]:
                    combinations.append({
                        'items': [top, bottom, layer],
                        'type': 'fusion_layered',
                        'structure': 'ethnic_top + western_bottom + layer'
                    })

        return combinations
    
    def score_outfit(self, outfit: Dict, preferences: Dict) -> OutfitScore:
        """Comprehensive outfit scoring"""
        items = outfit['items']
        
        # 1. Visual Coherence (CLIP embeddings)
        if len(items) == 1:
            visual_score = 1.0
        else:
            indices = [item['embedding_index'] for item in items]
            outfit_embeddings = self.embeddings[indices]
            similarities = cosine_similarity(outfit_embeddings)
            n = len(items)
            visual_score = (similarities.sum() - n) / (n * (n - 1))
        
        # 2. Color Harmony
        colors = [
            item.get('classification', {}).get('color_primary', '')
            for item in items
        ]
        
        if len(colors) <= 1:
            color_score = 1.0
        else:
            harmony_scores = []
            for i in range(len(colors)):
                for j in range(i + 1, len(colors)):
                    _, score = ColorHarmony.are_harmonious(colors[i], colors[j])
                    harmony_scores.append(score)
            color_score = np.mean(harmony_scores) if harmony_scores else 0.5
        
        # 3. Style Compatibility
        styles = [
            item.get('classification', {}).get('style', '')
            for item in items
        ]
        
        if len(styles) <= 1:
            style_score = 1.0
        else:
            style_scores = []
            for i in range(len(styles)):
                for j in range(i + 1, len(styles)):
                    _, score = StyleRules.are_compatible(styles[i], styles[j])
                    style_scores.append(score)
            style_score = np.mean(style_scores) if style_scores else 0.5
        
        # 4. Occasion Fit
        target_occasion = preferences.get('occasion', 'any')
        if target_occasion == 'any':
            occasion_score = 1.0
        else:
            occasion_matches = []
            for item in items:
                occasions = item.get('classification', {}).get('occasions', [])
                if not isinstance(occasions, list):
                    occasions = [occasions]
                match = any(target_occasion.lower() in str(occ).lower() for occ in occasions)
                occasion_matches.append(1.0 if match else 0.5)
            occasion_score = np.mean(occasion_matches)
        
        # 5. Overall Score (weighted)
        overall = (
            0.35 * visual_score +
            0.30 * color_score +
            0.20 * style_score +
            0.15 * occasion_score
        )
        
        return OutfitScore(
            visual_coherence=visual_score,
            color_harmony=color_score,
            style_compatibility=style_score,
            occasion_fit=occasion_score,
            overall=overall
        )
    
    def recommend_outfits(
        self,
        occasion: str = 'any',
        weather: str = 'any',
        style: str = 'any',
        gender: str = 'any',
        formality: str = 'any',
        num_outfits: int = 10,
        min_score: float = 0.5
    ) -> List[Dict]:
        """
        Generate smart outfit recommendations
        
        Returns:
            List of ranked outfit recommendations with scores
        """
        logger.info(f"Filtering: occasion={occasion}, weather={weather}, style={style}")
        
        # Filter items
        filtered = self.filter_by_preferences(
            occasion=occasion,
            weather=weather,
            style=style,
            gender=gender,
            formality=formality
        )
        
        total_items = sum(len(items) for items in filtered.values())
        logger.info(f"✅ Found {total_items} matching items")
        
        if total_items < 2:
            logger.warning("Not enough items for outfits")
            return []
        
        # Generate combinations
        logger.info("🎨 Generating combinations...")
        combinations = self.generate_combinations(filtered)
        logger.info(f"✅ Generated {len(combinations)} combinations")
        
        if not combinations:
            return []
        
        # Score all outfits
        logger.info("📊 Scoring outfits...")
        preferences = {
            'occasion': occasion,
            'weather': weather,
            'style': style
        }
        
        for combo in combinations:
            score = self.score_outfit(combo, preferences)
            combo['score'] = score
            combo['score_dict'] = score.to_dict()
        
        # Filter by minimum score
        valid_outfits = [
            combo for combo in combinations
            if combo['score'].overall >= min_score
        ]
        
        logger.info(f"✅ {len(valid_outfits)} outfits above threshold")
        
        # Sort by overall score
        valid_outfits.sort(key=lambda x: x['score'].overall, reverse=True)
        
        # Return top N
        top_outfits = valid_outfits[:num_outfits]
        logger.info(f"✅ Returning top {len(top_outfits)} outfits")
        
        return top_outfits


def main():
    """Test the generator"""
    
    print("="*80)
    print("ADVANCED OUTFIT RECOMMENDATION ENGINE")
    print("="*80)
    # generator = OutfitGenerator(
    #     metadata_jsonl="data/men_wardrobe_with_embeddings.jsonl",
    #     embeddings_npy="data/men_wardrobe_embeddings.npy"
    # )    
    # Initialize
    generator = OutfitGenerator(
        metadata_jsonl="data/women_wardrobe_with_embeddings.jsonl",
        embeddings_npy="data/women_wardrobe_embeddings.npy"
    )
    
    # Test different scenarios
    test_scenarios = [
        {
            'name': 'Casual Daily Wear',
            'occasion': 'casual',
            'style': 'any',
            'weather': 'any'
        },
        {
            'name': 'Formal Traditional',
            'occasion': 'formal',
            'style': 'traditional',
            'weather': 'any'
        },
        {
            'name': 'Office Wear',
            'occasion': 'office',
            'style': 'contemporary',
            'weather': 'any'
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'='*80}")
        
        outfits = generator.recommend_outfits(
            occasion=scenario['occasion'],
            style=scenario['style'],
            weather=scenario['weather'],
            num_outfits=3
        )
        
        if outfits:
            for i, outfit in enumerate(outfits, 1):
                print(f"\n🌟 OUTFIT #{i}")
                print(f"Type: {outfit['type']}")
                print(f"Score: {outfit['score'].overall:.1%}")
                print(f"  Visual: {outfit['score'].visual_coherence:.1%} | "
                      f"Color: {outfit['score'].color_harmony:.1%} | "
                      f"Style: {outfit['score'].style_compatibility:.1%}")
                print("Items:")
                for item in outfit['items']:
                    c = item.get('classification', {})
                    print(f"  • {c.get('category')}: {c.get('color_primary')} "
                          f"{c.get('specific_type')} ({c.get('style')})")
                print("-" * 40)
        else:
            print("❌ No outfits found for this scenario")
    
    print("\n🎉 Testing complete!")


if __name__ == "__main__":
    main()