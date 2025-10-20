# 👗 AI Fashion Stylist - Advanced Outfit Recommendation System

An intelligent outfit recommendation system using computer vision, color theory, and fashion rules to create perfectly matched outfits from your wardrobe.

## 🌟 Key Features

### Core Capabilities
- **Visual Harmony Analysis**: Uses Marqo-FashionSigLIP embeddings to measure visual coherence
- **Color Theory Integration**: Applies real color harmony rules (complementary, analogous, neutral matching)
- **Style Compatibility**: Smart matching of traditional, contemporary, fusion, and western styles
- **Multi-Factor Scoring**: Combines visual, color, style, and occasion fit scores
- **Resume Capability**: Crash-safe processing with automatic resume from interruption
- **Smart Filtering**: Filter by occasion, weather, style, formality, and gender
- **Advanced UI**: Modern Streamlit interface with favorites, sorting, and detailed scores

### Improvements Over Original

#### 1. Robust Embedding Generation
- ✅ Image validation (size, format, corruption check)
- ✅ Resume from interruption with hash-based change detection
- ✅ Incremental saving for crash safety
- ✅ Comprehensive error handling and logging
- ✅ Progress tracking with detailed statistics

#### 2. Intelligent Outfit Recommendation
- ✅ Color harmony rules based on color theory
- ✅ Style compatibility matrix for fashion rules
- ✅ Multi-dimensional scoring (visual + color + style + occasion)
- ✅ Smart combination generation with performance optimization
- ✅ Flexible filtering with multiple criteria
- ✅ Better category organization (ethnic/western separation)

#### 3. Enhanced User Interface
- ✅ Modern, responsive design with custom CSS
- ✅ Score breakdown (visual, color, style, occasion)
- ✅ Image gallery with item details
- ✅ Favorites system
- ✅ Sorting options (by different score types)
- ✅ Wardrobe statistics dashboard
- ✅ Error handling with user-friendly messages

## 📋 Requirements

```txt
# requirements.txt
torch>=2.0.0
open-clip-torch>=2.20.0
numpy>=1.24.0
scikit-learn>=1.3.0
Pillow>=10.0.0
tqdm>=4.65.0
streamlit>=1.28.0
pandas>=2.0.0
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo>
cd fashion-stylist

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Your wardrobe data should be in JSONL format:

```json
{
  "filename": "item.jpg",
  "image_path": "path/to/item.jpg",
  "classification": {
    "category": "men_kurta",
    "color_primary": "maroon",
    "color_secondary": ["gold"],
    "style": "traditional",
    "occasions": ["wedding", "festival"],
    "weather": ["all_season"],
    "formality": "formal",
    "gender": "male",
    ...
  }
}
```

Place your data in:
- `data/men_wardrobe_database.jsonl`
- `data/women_wardrobe_database.jsonl`

### 3. Generate Embeddings

```bash
python 1_generate_embeddings.py
```

**Features:**
- Automatically resumes if interrupted
- Validates all images before processing
- Saves checkpoints every 100 images
- Handles corrupted images gracefully

**Output:**
- `data/men_wardrobe_embeddings.npy` - 768-dim CLIP embeddings
- `data/men_wardrobe_with_embeddings.jsonl` - Metadata with indices

### 4. Test Recommendation Engine

```bash
python 2_outfit_generator.py
```

**Tests 3 scenarios:**
1. Casual daily wear
2. Formal traditional
3. Office wear

**Scoring metrics:**
- Visual coherence: 35%
- Color harmony: 30%
- Style compatibility: 20%
- Occasion fit: 15%

### 5. Launch Web Interface

```bash
streamlit run 3_streamlit_app.py
```

Access at: http://localhost:8501

## 🎨 Color Harmony Rules

The system understands color theory:

### Color Families
- **Neutral**: white, black, grey, beige, cream
- **Warm**: red, orange, yellow, maroon, mustard, brown
- **Cool**: blue, green, purple, navy, teal
- **Earth**: brown, olive, khaki, tan, camel

### Harmony Scores
- Same family: 0.9 (high harmony)
- Neutral + any: 0.85 (always works)
- Warm + earth: 0.8 (natural pairing)
- Cool colors: 0.75 (cohesive)
- Warm + cool: 0.4 (contrasting - use carefully)

## 💃 Style Compatibility

Fashion-aware style matching:

| Style | Compatible With |
|-------|----------------|
| Traditional | Traditional, Fusion, Festive |
| Contemporary | Contemporary, Fusion, Casual, Formal |
| Fusion | Traditional, Contemporary, Casual |
| Casual | Casual, Contemporary, Western |
| Formal | Formal, Contemporary, Traditional |
| Western | Western, Casual, Contemporary |

## 📊 Outfit Scoring System

Each outfit receives a comprehensive score:

```python
Overall Score = (
    0.35 × Visual Coherence +      # CLIP embedding similarity
    0.30 × Color Harmony +         # Color theory rules
    0.20 × Style Compatibility +   # Fashion style rules
    0.15 × Occasion Fit           # Matches your occasion
)
```

## 🔧 Advanced Configuration

### Customizing Recommendations

```python
outfits = generator.recommend_outfits(
    occasion='wedding',          # Filter by occasion
    weather='summer',            # Filter by season
    style='traditional',         # Filter by style
    gender='male',              # Filter by gender
    formality='formal',         # Filter by formality level
    num_outfits=10,            # Number to return
    min_score=0.6              # Quality threshold
)
```

### Category Organization

The system intelligently organizes items:

- **Ethnic Tops**: Kurtas, kurtis, anarkalis
- **Ethnic Bottoms**: Palazzo, churidar, salwar
- **Ethnic Full**: Saree, lehenga, sherwani
- **Western Tops**: Shirts, t-shirts
- **Western Bottoms**: Jeans, trousers, pants
- **Layers**: Jackets, dupattas, shawls

### Combination Types Generated

1. **Ethnic Separates**: Top + Bottom
2. **Ethnic Layered**: Top + Bottom + Layer
3. **Traditional Full**: Saree/Sherwani alone
4. **Traditional Layered**: Full outfit + Layer
5. **Western**: Shirt + Pants/Jeans

## 🐛 Troubleshooting

### Issue: "Not enough items for outfits"

**Solution:** Your filters are too restrictive. Try:
1. Use `style='any'` instead of specific style
2. Use `occasion='any'` for broader results
3. Check if your wardrobe has compatible items (e.g., tops AND bottoms)

### Issue: Embeddings generation fails

**Solution:**
```bash
# Install/update dependencies
pip install --upgrade open-clip-torch torch

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Images not displaying in Streamlit

**Solution:**
- Ensure image paths in JSONL are correct
- Use absolute paths or paths relative to script location
- Check image file permissions

## 📈 Performance Tips

1. **Large Wardrobes**: Limit items per category (done automatically)
2. **Faster Processing**: Use GPU for embeddings
3. **Memory Issues**: Process in batches using start/end parameters
4. **Better Results**: Ensure high-quality, well-lit product images

## 🎯 Future Enhancements

- [ ] Virtual try-on visualization
- [ ] Weather-based smart suggestions
- [ ] Trend analysis and seasonal recommendations
- [ ] Social sharing of favorite outfits
- [ ] Outfit history and analytics
- [ ] Color palette extraction from user photos
- [ ] Integration with wardrobe management apps
- [ ] Multi-language support

## 📝 Citation

If you use this system, please cite:

```bibtex
@software{ai_fashion_stylist_2025,
  title = {AI Fashion Stylist: Advanced Outfit Recommendation System},
  author = {Your Name},
  year = {2025},
  note = {Uses Marqo-FashionSigLIP embeddings}
}
```

## 🙏 Acknowledgments

- **Marqo-FashionSigLIP**: Fashion-specific CLIP embeddings
- **OpenCLIP**: Open-source CLIP implementation
- **Streamlit**: Beautiful web interface framework

## 📄 License

MIT License - Feel free to use and modify!

---

Made with ❤️ and AI • [Report Issues](https://github.com/yourusername/repo/issues)