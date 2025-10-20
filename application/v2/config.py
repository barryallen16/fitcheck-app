# config.py - Centralized configuration
"""
Configuration file for AI Fashion Stylist
Modify these settings to customize the system
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
EXPORTS_DIR = BASE_DIR / "exports"

# Create directories
for dir_path in [DATA_DIR, CHECKPOINTS_DIR, EXPORTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Embedding model
EMBEDDING_MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"
EMBEDDING_DIM = 768

# Alternative models (uncomment to use)
# EMBEDDING_MODEL = "hf-hub:Marqo/marqo-fashionCLIP"
# EMBEDDING_MODEL = "ViT-B-32"  # Standard CLIP

# Device settings
USE_GPU = True  # Set to False to force CPU
BATCH_SIZE = 1  # Increase if GPU memory allows

# ============================================================================
# PROCESSING CONFIGURATION
# ============================================================================

# Image validation
MAX_IMAGE_SIZE_MB = 50
MIN_IMAGE_SIZE_BYTES = 1024
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.webp']

# Checkpoint frequency
CHECKPOINT_EVERY_N = 100  # Save checkpoint every N images

# Processing limits
MAX_ITEMS_PER_CATEGORY = 100  # For performance in outfit generation

# ============================================================================
# OUTFIT SCORING WEIGHTS
# ============================================================================

SCORING_WEIGHTS = {
    'visual_coherence': 0.35,
    'color_harmony': 0.30,
    'style_compatibility': 0.20,
    'occasion_fit': 0.15
}

# Minimum score threshold
MIN_OUTFIT_SCORE = 0.5

# ============================================================================
# COLOR HARMONY RULES
# ============================================================================

COLOR_FAMILIES = {
    'neutral': ['white', 'black', 'grey', 'gray', 'beige', 'cream', 'ivory', 'off-white'],
    'warm': ['red', 'orange', 'yellow', 'maroon', 'mustard', 'gold', 'brown', 'rust', 'coral', 'peach'],
    'cool': ['blue', 'green', 'purple', 'navy', 'teal', 'turquoise', 'violet', 'indigo', 'cyan'],
    'earth': ['brown', 'olive', 'khaki', 'tan', 'camel', 'terracotta', 'sienna', 'umber']
}

# Harmony scores for different color family combinations
HARMONY_SCORES = {
    ('same', 'same'): 0.95,
    ('neutral', 'any'): 0.85,
    ('warm', 'earth'): 0.80,
    ('cool', 'cool'): 0.75,
    ('warm', 'cool'): 0.40,
    ('default',): 0.60
}

# ============================================================================
# STYLE COMPATIBILITY RULES
# ============================================================================

STYLE_COMPATIBILITY = {
    'traditional': ['traditional', 'fusion', 'festive'],
    'contemporary': ['contemporary', 'fusion', 'casual', 'formal'],
    'fusion': ['traditional', 'contemporary', 'fusion', 'casual'],
    'casual': ['casual', 'contemporary', 'western', 'fusion'],
    'formal': ['formal', 'contemporary', 'traditional'],
    'festive': ['festive', 'traditional', 'fusion'],
    'western': ['western', 'casual', 'contemporary']
}

# ============================================================================
# CATEGORY MAPPINGS
# ============================================================================

CATEGORY_MAPPING = {
    # Women's ethnic tops
    'women_kurta': 'ethnic_tops',
    'women_anarkali_kurta': 'ethnic_tops',
    'women_a_line_kurta': 'ethnic_tops',
    'kurti': 'ethnic_tops',
    'blouse': 'ethnic_tops',
    'choli': 'ethnic_tops',
    
    # Men's ethnic tops
    'men_kurta': 'ethnic_tops',
    'kurta': 'ethnic_tops',
    
    # Ethnic bottoms
    'palazzo': 'ethnic_bottoms',
    'leggings_salwar': 'ethnic_bottoms',
    'churidar': 'ethnic_bottoms',
    'salwar': 'ethnic_bottoms',
    'dhoti_pants': 'ethnic_bottoms',
    
    # Full ethnic outfits
    'saree': 'ethnic_full',
    'lehenga': 'ethnic_full',
    'sherwani': 'ethnic_full',
    'gown': 'ethnic_full',
    
    # Western tops
    'shirt': 'western_tops',
    't_shirt': 'western_tops',
    'top': 'western_tops',
    
    # Western bottoms
    'jeans': 'western_bottoms',
    'trousers': 'western_bottoms',
    'pants': 'western_bottoms',
    
    # Layers
    'nehru_jacket': 'layers',
    'jacket': 'layers',
    'dupatta': 'layers',
    'shawl': 'layers',
    'cape': 'layers'
}

# ============================================================================
# UI CONFIGURATION
# ============================================================================

# Streamlit settings
PAGE_TITLE = "AI Fashion Stylist"
PAGE_ICON = "👗"
LAYOUT = "wide"

# Theme colors
PRIMARY_COLOR = "#667eea"
SECONDARY_COLOR = "#764ba2"
ACCENT_COLOR = "#10b981"

# Display settings
DEFAULT_NUM_OUTFITS = 10
MAX_NUM_OUTFITS = 20
IMAGE_WIDTH = 300

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = DATA_DIR / "fashion_stylist.log"

# ============================================================================
# WARDROBE CONFIGURATIONS
# ============================================================================

WARDROBES = {
    'men': {
        'input': DATA_DIR / 'men_wardrobe_database.jsonl',
        'embeddings': DATA_DIR / 'men_wardrobe_embeddings.npy',
        'metadata': DATA_DIR / 'men_wardrobe_with_embeddings.jsonl'
    },
    'women': {
        'input': DATA_DIR / 'women_wardrobe_database.jsonl',
        'embeddings': DATA_DIR / 'women_wardrobe_embeddings.npy',
        'metadata': DATA_DIR / 'women_wardrobe_with_embeddings.jsonl'
    }
}

# ============================================================================
# EXPORT SETTINGS
# ============================================================================

EXPORT_FORMATS = ['json', 'csv', 'pdf']
INCLUDE_IMAGES_IN_EXPORT = True
IMAGE_EXPORT_SIZE = (400, 400)  # Width, Height

# ============================================================================
# ADVANCED FEATURES (Optional)
# ============================================================================

# Enable experimental features
ENABLE_COLOR_EXTRACTION = False  # Extract dominant colors from images
ENABLE_PATTERN_DETECTION = False  # Detect patterns automatically
ENABLE_TREND_ANALYSIS = False     # Analyze fashion trends

# API settings (if integrating with external services)
API_ENABLED = False
API_KEY = os.getenv('FASHION_API_KEY', '')
API_ENDPOINT = os.getenv('FASHION_API_ENDPOINT', '')

# ============================================================================
# VALIDATION RULES
# ============================================================================

REQUIRED_FIELDS = [
    'image_path',
    'classification.category',
    'classification.color_primary',
    'classification.style'
]

VALID_CATEGORIES = list(CATEGORY_MAPPING.keys())
VALID_STYLES = list(STYLE_COMPATIBILITY.keys())
VALID_OCCASIONS = ['casual', 'formal', 'office', 'party', 'wedding', 'festival', 'daily_wear']
VALID_WEATHER = ['summer', 'winter', 'monsoon', 'all_season']
VALID_FORMALITY = ['casual', 'semi_formal', 'formal', 'festive']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_wardrobe_config(wardrobe_type: str) -> dict:
    """Get configuration for specific wardrobe type"""
    return WARDROBES.get(wardrobe_type.lower(), WARDROBES['men'])

def get_device():
    """Get computation device"""
    if USE_GPU:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cpu'

def validate_config():
    """Validate configuration settings"""
    issues = []
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        issues.append(f"Data directory not found: {DATA_DIR}")
    
    # Check scoring weights sum to 1.0
    weight_sum = sum(SCORING_WEIGHTS.values())
    if not (0.99 <= weight_sum <= 1.01):
        issues.append(f"Scoring weights must sum to 1.0, currently: {weight_sum}")
    
    return issues

# Validate on import
_validation_issues = validate_config()
if _validation_issues:
    print("⚠️  Configuration warnings:")
    for issue in _validation_issues:
        print(f"  - {issue}")