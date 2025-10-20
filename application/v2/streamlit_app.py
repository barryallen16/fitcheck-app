# 3_streamlit_app.py - Improved Version
# Modern web interface with enhanced features

import streamlit as st
from PIL import Image
import json
from pathlib import Path
import sys

# Import outfit generator
sys.path.append('.')
from outfit_generator import OutfitGenerator

# Page config
st.set_page_config(
    page_title="AI Fashion Stylist",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .outfit-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .score-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.2rem;
    }
    .score-excellent {
        background: #10b981;
        color: white;
    }
    .score-good {
        background: #3b82f6;
        color: white;
    }
    .score-fair {
        background: #f59e0b;
        color: white;
    }
    .item-tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 15px;
        background: #f3f4f6;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'outfits' not in st.session_state:
    st.session_state.outfits = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

# Load generator
@st.cache_resource
def load_generator(wardrobe_type='men'):
    """Load outfit generator with caching"""
    try:
        if wardrobe_type == 'men':
            metadata = "data/men_wardrobe_with_embeddings.jsonl"
            embeddings = "data/men_wardrobe_embeddings.npy"
        else:
            metadata = "data/women_wardrobe_with_embeddings.jsonl"
            embeddings = "data/women_wardrobe_embeddings.npy"
        
        generator = OutfitGenerator(
            metadata_jsonl=metadata,
            embeddings_npy=embeddings
        )
        return generator, None
    except Exception as e:
        return None, str(e)

# Header
st.markdown('<p class="main-header">👗 AI Fashion Stylist</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6b7280; font-size: 1.1rem;">Smart Outfit Recommendations Powered by AI</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Your Preferences")
    
    # Wardrobe selection
    wardrobe_type = st.radio(
        "👤 Wardrobe",
        ["Men's", "Women's"],
        index=0,
        horizontal=True
    )
    
    wardrobe_key = 'men' if wardrobe_type == "Men's" else 'women'
    
    # Load generator
    generator, error = load_generator(wardrobe_key)
    
    if error:
        st.error(f"❌ Error loading wardrobe: {error}")
        st.info("💡 Make sure you've run 1_generate_embeddings.py first!")
        st.stop()
    
    st.session_state.generator = generator
    
    st.markdown("---")
    
    # Filters
    st.subheader("🎯 Filters")
    
    occasion = st.selectbox(
        "🎉 Occasion",
        ["any", "casual", "formal", "office", "party", "wedding", "festival", "daily_wear"],
        index=0
    )
    
    weather = st.selectbox(
        "🌤️ Weather",
        ["any", "summer", "winter", "monsoon"],  # <-- REMOVED 'all_season'
        index=0
    )
    
    style = st.selectbox(
        "💃 Style",
        ["any", "traditional", "contemporary", "fusion", "casual", "formal", "festive", "western"],
        index=0
    )
    
    formality = st.selectbox(
        "👔 Formality",
        ["any", "casual", "semi_formal", "formal", "festive"],
        index=0
    )
    
    st.markdown("---")
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        num_outfits = st.slider(
            "Number of recommendations",
            min_value=3,
            max_value=20,
            value=10
        )
        
        min_score = st.slider(
            "Minimum quality score",
            min_value=0.3,
            max_value=0.9,
            value=0.5,
            step=0.05
        )
    
    st.markdown("---")
    
    # Generate button
    generate_button = st.button(
        "✨ Generate Outfits",
        type="primary",
        use_container_width=True
    )

# Main content area
if generate_button:
    with st.spinner("🤖 AI is creating perfect outfits for you..."):
        outfits = generator.recommend_outfits(
            occasion=occasion,
            weather=weather,
            style=style,
            formality=formality,
            num_outfits=num_outfits,
            min_score=min_score
        )
        
        st.session_state.outfits = outfits
    
    if not outfits:
        st.warning("😕 No outfits found matching your criteria. Try adjusting the filters!")
    else:
        st.success(f"✨ Generated {len(outfits)} amazing outfit recommendations!")

# Display outfits
if st.session_state.outfits:
    st.markdown("## 🌟 Your Personalized Outfits")
    
    # Add sorting options
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            ["Overall Score", "Visual Harmony", "Color Harmony", "Style Match"]
        )
    
    # Sort outfits
    if sort_by == "Overall Score":
        sorted_outfits = sorted(st.session_state.outfits, 
                               key=lambda x: x['score'].overall, reverse=True)
    elif sort_by == "Visual Harmony":
        sorted_outfits = sorted(st.session_state.outfits, 
                               key=lambda x: x['score'].visual_coherence, reverse=True)
    elif sort_by == "Color Harmony":
        sorted_outfits = sorted(st.session_state.outfits, 
                               key=lambda x: x['score'].color_harmony, reverse=True)
    else:
        sorted_outfits = sorted(st.session_state.outfits, 
                               key=lambda x: x['score'].style_compatibility, reverse=True)
    
    # Display each outfit
    for i, outfit in enumerate(sorted_outfits, 1):
        score = outfit['score']
        
        # Overall score badge
        if score.overall >= 0.8:
            score_class = "score-excellent"
            score_label = "Excellent Match"
        elif score.overall >= 0.6:
            score_class = "score-good"
            score_label = "Good Match"
        else:
            score_class = "score-fair"
            score_label = "Fair Match"
        
        st.markdown(f"### 🎯 Outfit #{i}")
        
        # Score badges
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f'<span class="score-badge {score_class}">{score_label} ({score.overall:.0%})</span>', 
                       unsafe_allow_html=True)
        with col2:
            fav_button = st.button("❤️ Save", key=f"fav_{i}")
            if fav_button:
                st.session_state.favorites.append(outfit)
                st.success("Saved to favorites!")
        
        # Detailed scores
        score_col1, score_col2, score_col3, score_col4 = st.columns(4)
        with score_col1:
            st.metric("👁️ Visual", f"{score.visual_coherence:.0%}")
        with score_col2:
            st.metric("🎨 Color", f"{score.color_harmony:.0%}")
        with score_col3:
            st.metric("✨ Style", f"{score.style_compatibility:.0%}")
        with score_col4:
            st.metric("🎯 Occasion", f"{score.occasion_fit:.0%}")
        
        # Display items
        item_cols = st.columns(len(outfit['items']))
        
        for col, item in zip(item_cols, outfit['items']):
            with col:
                classification = item.get('classification', {})
                
                # Display image
                try:
                    img_path = item.get('image_path', '')
                    if Path(img_path).exists():
                        img = Image.open(img_path)
                        st.image(img, width='stretch')
                    else:
                        st.info("🖼️ Image not found")
                except Exception as e:
                    st.warning(f"Error loading image: {e}")
                
                # Item details
                st.markdown(f"**{classification.get('category', 'Item').replace('_', ' ').title()}**")
                st.markdown(f"<span class='item-tag'>🎨 {classification.get('color_primary', 'N/A')}</span>", 
                           unsafe_allow_html=True)
                st.markdown(f"<span class='item-tag'>✨ {classification.get('style', 'N/A')}</span>", 
                           unsafe_allow_html=True)
                
                # Expandable details
                with st.expander("View details"):
                    st.write(f"**Type:** {classification.get('specific_type', 'N/A')}")
                    st.write(f"**Material:** {classification.get('material', 'N/A')}")
                    st.write(f"**Pattern:** {classification.get('pattern', 'N/A')}")
                    st.write(f"**Fit:** {classification.get('fit', 'N/A')}")
                    
                    occasions = classification.get('occasions', [])
                    if occasions:
                        st.write(f"**Best for:** {', '.join(occasions[:3])}")
        
        st.markdown("---")

else:
    # Welcome screen
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### 👈 Set Your Style\nChoose your preferences from the sidebar")
    
    with col2:
        st.info("### ✨ Get Matched\nClick 'Generate Outfits' to see recommendations")
    
    with col3:
        st.info("### 💾 Save Favorites\nSave outfits you love for quick access")
    
    if generator:
        st.markdown("### 📊 Your Wardrobe Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Items", len(generator.wardrobe))
        
        with col2:
            tops = (len(generator.items_by_type.get('ethnic_tops', [])) + 
                   len(generator.items_by_type.get('western_tops', [])))
            st.metric("Tops", tops)
        
        with col3:
            bottoms = (len(generator.items_by_type.get('ethnic_bottoms', [])) + 
                      len(generator.items_by_type.get('western_bottoms', [])))
            st.metric("Bottoms", bottoms)
        
        with col4:
            full = len(generator.items_by_type.get('ethnic_full', []))
            st.metric("Complete Outfits", full)
        
        # Category breakdown
        st.markdown("### 📦 Categories")
        for cat_type, items in generator.items_by_type.items():
            if items:
                st.markdown(f"**{cat_type.replace('_', ' ').title()}:** {len(items)} items")

# Favorites section
if st.session_state.favorites:
    st.markdown("---")
    st.markdown("## ❤️ Your Favorites")
    
    for i, outfit in enumerate(st.session_state.favorites, 1):
        st.markdown(f"**Favorite #{i}** - {outfit['type'].replace('_', ' ').title()}")
        
        cols = st.columns(len(outfit['items']) + 1)
        for col, item in zip(cols, outfit['items']):
            with col:
                try:
                    img = Image.open(item.get('image_path', ''))
                    st.image(img, use_container_width=True)
                except:
                    st.write("🖼️")
        
        with cols[-1]:
            if st.button("🗑️ Remove", key=f"remove_fav_{i}"):
                st.session_state.favorites.pop(i-1)
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem 0;'>
    <p>Made with ❤️ using AI • Powered by Marqo-FashionSigLIP</p>
    <p style='font-size: 0.85rem;'>Smart fashion recommendations based on visual harmony, color theory, and style compatibility</p>
</div>
""", unsafe_allow_html=True)