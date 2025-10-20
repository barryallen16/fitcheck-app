# 3_streamlit_app.py
# Web interface for outfit recommendations

import streamlit as st
from PIL import Image
import pandas as pd
import json
from outfit_generator import OutfitGenerator

# Page config
st.set_page_config(
    page_title="AI Fashion Recommendation",
    page_icon="👗",
    layout="wide"
)

# Title
st.title("👗 AI Fashion Outfit Recommendation System")
st.markdown("*Powered by Marqo-FashionSigLIP + AI*")
st.markdown("---")

# Initialize generator
@st.cache_resource
def load_generator():
    try:
        generator = OutfitGenerator(
            metadata_jsonl="data/men/men_wardrobe_with_embeddings.jsonl",
            embeddings_npy="data/men/men_wardrobe_embeddings.npy"
        )
        return generator
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure you've run 1_generate_embeddings.py first!")
        return None

generator = load_generator()

if generator is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Preferences")
    
    occasion = st.selectbox(
        "🎉 Occasion",
        ["any", "wedding", "festival", "party", "casual", "office", "daily_wear"]
    )
    
    weather = st.selectbox(
        "🌤️ Weather",
        ["any", "summer", "winter", "monsoon", "all_season"]
    )
    
    style = st.selectbox(
        "💃 Style",
        ["any", "traditional", "contemporary", "fusion", "casual", "formal"]
    )
    
    num_outfits = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5
    )
    
    generate_button = st.button("✨ Generate Outfits", type="primary", use_container_width=True)

# Main area
if generate_button:
    
    with st.spinner("🤖 AI is creating perfect outfits for you..."):
        
        outfits = generator.recommend_outfits(
            occasion=occasion,
            weather=weather,
            style=style,
            num_outfits=num_outfits
        )
        
        if not outfits:
            st.warning("No outfits found matching your criteria. Try adjusting filters!")
        else:
            st.success(f"✅ Generated {len(outfits)} outfit recommendations!")
            
            st.markdown("---")
            st.markdown("## 👗 Your Outfit Recommendations")
            
            for i, outfit in enumerate(outfits, 1):
                
                st.markdown(f"### 🌟 Outfit #{i}: {outfit['type'].replace('_', ' ').title()}")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Display items
                    cols = st.columns(len(outfit['items']))
                    
                    for col, item in zip(cols, outfit['items']):
                        with col:
                            classification = item.get('classification', {})
                            
                            # Try to display image
                            try:
                                img = Image.open(item.get('image_path', ''))
                                st.image(img, use_container_width=True)
                            except:
                                st.write("🖼️ Image")
                            
                            st.caption(f"**{classification.get('category', 'Item')}**")
                            st.caption(f"Color: {classification.get('color_primary', 'N/A')}")
                            st.caption(f"Type: {classification.get('specific_type', 'N/A')}")
                
                with col2:
                    st.metric("Visual Harmony", f"{outfit['visual_coherence']:.0%}")
                    st.caption(f"Structure: {outfit['structure']}")
                    st.caption(f"Type: {outfit['type'].replace('_', ' ').title()}")
                
                st.markdown("---")

else:
    # Welcome screen
    st.info("👈 Set your preferences and click **Generate Outfits** to get started!")
    
    st.markdown("### 📊 Wardrobe Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Items", len(generator.wardrobe))
    
    with col2:
        tops_count = len(generator.items_by_type.get('tops', [])) + len(generator.items_by_type.get('western_tops', []))
        st.metric("Tops", tops_count)
    
    with col3:
        bottoms_count = len(generator.items_by_type.get('bottoms', [])) + len(generator.items_by_type.get('western_bottoms', []))
        st.metric("Bottoms", bottoms_count)
