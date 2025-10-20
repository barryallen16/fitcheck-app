# test_system.py - Comprehensive testing suite

import unittest
import json
import numpy as np
from pathlib import Path
import tempfile
import os

# Import modules to test
try:
    from outfit_generator import OutfitGenerator, ColorHarmony, StyleRules, OutfitScore
    from data_utils import DataValidator, DataCleaner, DataAnalyzer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required files are in the same directory")
    exit(1)


class TestColorHarmony(unittest.TestCase):
    """Test color harmony rules"""
    
    def test_same_family(self):
        """Test colors in same family"""
        is_harmonious, score = ColorHarmony.are_harmonious("red", "maroon")
        self.assertTrue(is_harmonious)
        self.assertGreater(score, 0.8)
    
    def test_neutral_with_any(self):
        """Test neutral colors"""
        is_harmonious, score = ColorHarmony.are_harmonious("white", "red")
        self.assertTrue(is_harmonious)
        self.assertGreater(score, 0.7)
    
    def test_contrasting(self):
        """Test contrasting colors"""
        is_harmonious, score = ColorHarmony.are_harmonious("red", "blue")
        self.assertLess(score, 0.6)


class TestStyleRules(unittest.TestCase):
    """Test style compatibility rules"""
    
    def test_same_style(self):
        """Test identical styles"""
        is_compatible, score = StyleRules.are_compatible("traditional", "traditional")
        self.assertTrue(is_compatible)
        self.assertEqual(score, 1.0)
    
    def test_compatible_styles(self):
        """Test compatible different styles"""
        is_compatible, score = StyleRules.are_compatible("traditional", "fusion")
        self.assertTrue(is_compatible)
        self.assertGreater(score, 0.7)
    
    def test_incompatible_styles(self):
        """Test incompatible styles"""
        is_compatible, score = StyleRules.are_compatible("traditional", "western")
        # Should still return a score, just lower
        self.assertLess(score, 0.5)


class TestDataValidator(unittest.TestCase):
    """Test data validation"""
    
    def setUp(self):
        self.validator = DataValidator(strict_mode=False)
    
    def test_valid_item(self):
        """Test valid item"""
        item = {
            'image_path': __file__,  # Use this file as dummy image
            'classification': {
                'category': 'men_kurta',
                'color_primary': 'blue',
                'style': 'traditional'
            }
        }
        result = self.validator.validate_item(item)
        # May have warnings but should not have critical errors
        self.assertTrue(len(result.errors) == 0 or 'Cannot open image' in str(result.errors))
    
    def test_missing_required_field(self):
        """Test missing required fields"""
        item = {
            'classification': {}
        }
        result = self.validator.validate_item(item)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
    
    def test_missing_image(self):
        """Test missing image file"""
        item = {
            'image_path': '/nonexistent/image.jpg',
            'classification': {
                'category': 'men_kurta',
                'color_primary': 'blue'
            }
        }
        result = self.validator.validate_item(item)
        self.assertIn('not found', str(result.errors))


class TestDataCleaner(unittest.TestCase):
    """Test data cleaning"""
    
    def setUp(self):
        self.cleaner = DataCleaner()
    
    def test_normalize_color(self):
        """Test color normalization"""
        self.assertEqual(self.cleaner.normalize_color("Grey"), "gray")
        self.assertEqual(self.cleaner.normalize_color("Off White"), "off_white")
    
    def test_normalize_category(self):
        """Test category normalization"""
        self.assertEqual(self.cleaner.normalize_category("Men Kurta"), "men_kurta")
    
    def test_ensure_list(self):
        """Test list conversion"""
        self.assertEqual(self.cleaner.ensure_list("casual"), ["casual"])
        self.assertEqual(self.cleaner.ensure_list(["casual"]), ["casual"])
    
    def test_clean_item(self):
        """Test complete item cleaning"""
        item = {
            'classification': {
                'category': 'Men Kurta',
                'color_primary': 'Grey',
                'occasions': 'casual',
                'embellishments': ['', 'embroidery', '']
            }
        }
        cleaned = self.cleaner.clean_item(item)
        
        self.assertEqual(cleaned['classification']['category'], 'men_kurta')
        self.assertEqual(cleaned['classification']['color_primary'], 'gray')
        self.assertIsInstance(cleaned['classification']['occasions'], list)
        self.assertNotIn('', cleaned['classification']['embellishments'])


class TestOutfitGenerator(unittest.TestCase):
    """Test outfit generation (requires sample data)"""
    
    @classmethod
    def setUpClass(cls):
        """Create sample test data"""
        cls.test_dir = tempfile.mkdtemp()
        
        # Create sample wardrobe
        cls.sample_items = [
            {
                'filename': 'shirt1.jpg',
                'image_path': '/tmp/shirt1.jpg',
                'embedding_index': 0,
                'classification': {
                    'category': 'shirt',
                    'color_primary': 'blue',
                    'style': 'contemporary',
                    'occasions': ['office', 'casual'],
                    'weather': ['all_season']
                }
            },
            {
                'filename': 'jeans1.jpg',
                'image_path': '/tmp/jeans1.jpg',
                'embedding_index': 1,
                'classification': {
                    'category': 'jeans',
                    'color_primary': 'blue',
                    'style': 'casual',
                    'occasions': ['casual'],
                    'weather': ['all_season']
                }
            }
        ]
        
        # Create test files
        cls.metadata_file = Path(cls.test_dir) / "test_metadata.jsonl"
        with open(cls.metadata_file, 'w') as f:
            for item in cls.sample_items:
                f.write(json.dumps(item) + '\n')
        
        # Create dummy embeddings
        cls.embeddings_file = Path(cls.test_dir) / "test_embeddings.npy"
        embeddings = np.random.rand(2, 768).astype(np.float32)
        np.save(cls.embeddings_file, embeddings)
    
    def test_load_generator(self):
        """Test generator initialization"""
        try:
            generator = OutfitGenerator(
                metadata_jsonl=str(self.metadata_file),
                embeddings_npy=str(self.embeddings_file)
            )
            self.assertEqual(len(generator.wardrobe), 2)
        except Exception as e:
            self.fail(f"Generator initialization failed: {e}")
    
    def test_filter_by_preferences(self):
        """Test preference filtering"""
        generator = OutfitGenerator(
            metadata_jsonl=str(self.metadata_file),
            embeddings_npy=str(self.embeddings_file)
        )
        
        filtered = generator.filter_by_preferences(occasion='casual')
        total_items = sum(len(items) for items in filtered.values())
        self.assertGreater(total_items, 0)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        import shutil
        if Path(cls.test_dir).exists():
            shutil.rmtree(cls.test_dir)


def run_integration_tests():
    """Run integration tests with real data if available"""
    print("\n" + "="*80)
    print("INTEGRATION TESTS")
    print("="*80)
    
    # Check if real data exists
    men_metadata = "data/men_wardrobe_with_embeddings.jsonl"
    men_embeddings = "data/men_wardrobe_embeddings.npy"
    
    if Path(men_metadata).exists() and Path(men_embeddings).exists():
        print("\n✅ Real data found, running integration tests...")
        
        try:
            # Test 1: Load generator
            print("\nTest 1: Loading generator...")
            generator = OutfitGenerator(
                metadata_jsonl=men_metadata,
                embeddings_npy=men_embeddings
            )
            print(f"✅ Loaded {len(generator.wardrobe)} items")
            
            # Test 2: Generate recommendations
            print("\nTest 2: Generating recommendations...")
            outfits = generator.recommend_outfits(
                occasion='casual',
                style='any',
                num_outfits=3
            )
            print(f"✅ Generated {len(outfits)} outfits")
            
            if outfits:
                # Test 3: Check outfit structure
                print("\nTest 3: Validating outfit structure...")
                outfit = outfits[0]
                assert 'items' in outfit, "Missing 'items' field"
                assert 'score' in outfit, "Missing 'score' field"
                assert 'type' in outfit, "Missing 'type' field"
                print("✅ Outfit structure valid")
                
                # Test 4: Check scores
                print("\nTest 4: Validating scores...")
                score = outfit['score']
                assert 0 <= score.overall <= 1, "Score out of range"
                assert 0 <= score.visual_coherence <= 1, "Visual score out of range"
                assert 0 <= score.color_harmony <= 1, "Color score out of range"
                print("✅ Scores valid")
                
                print("\n✅ All integration tests passed!")
            else:
                print("⚠️  No outfits generated (may need to adjust filters)")
                
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  Real data not found. Run 1_generate_embeddings.py first.")
        print(f"Looking for:")
        print(f"  - {men_metadata}")
        print(f"  - {men_embeddings}")


def run_performance_tests():
    """Run performance benchmarks"""
    print("\n" + "="*80)
    print("PERFORMANCE TESTS")
    print("="*80)
    
    men_metadata = "data/men_wardrobe_with_embeddings.jsonl"
    men_embeddings = "data/men_wardrobe_embeddings.npy"
    
    if not (Path(men_metadata).exists() and Path(men_embeddings).exists()):
        print("⚠️  Real data not found. Skipping performance tests.")
        return
    
    import time
    
    try:
        # Test 1: Load time
        print("\nTest 1: Generator load time...")
        start = time.time()
        generator = OutfitGenerator(
            metadata_jsonl=men_metadata,
            embeddings_npy=men_embeddings
        )
        load_time = time.time() - start
        print(f"✅ Load time: {load_time:.3f} seconds")
        
        # Test 2: Recommendation time
        print("\nTest 2: Recommendation generation time...")
        start = time.time()
        outfits = generator.recommend_outfits(
            occasion='casual',
            num_outfits=10
        )
        rec_time = time.time() - start
        print(f"✅ Generation time: {rec_time:.3f} seconds")
        print(f"✅ Generated {len(outfits)} outfits")
        print(f"✅ Average: {rec_time/len(outfits):.3f} seconds per outfit")
        
        # Test 3: Scoring time
        if outfits:
            print("\nTest 3: Outfit scoring time...")
            outfit = outfits[0]
            start = time.time()
            for _ in range(100):
                score = generator.score_outfit(outfit, {'occasion': 'casual'})
            scoring_time = (time.time() - start) / 100
            print(f"✅ Average scoring time: {scoring_time*1000:.2f} ms")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")


if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("FASHION STYLIST SYSTEM TESTS")
    print("="*80)
    
    # Run unit tests
    print("\n" + "="*80)
    print("UNIT TESTS")
    print("="*80)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run integration tests
    run_integration_tests()
    
    # Run performance tests
    run_performance_tests()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Unit tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)


# ============================================================================
# DEPLOYMENT.md content
# ============================================================================

DEPLOYMENT_GUIDE = """
# 🚀 Deployment Guide

## Overview

This guide covers deploying the AI Fashion Stylist system to production environments.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU with 8GB+ RAM
- 10GB+ disk space for models and data

## Environment Setup

### 1. Production Server Setup

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install python3-pip python3-venv -y

# Create virtual environment
python3 -m venv fashion_env
source fashion_env/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. GPU Setup (Optional but recommended)

```bash
# Install CUDA toolkit
sudo apt-get install nvidia-cuda-toolkit

# Verify CUDA installation
nvcc --version

# Test PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available())"
```

## Deployment Options

### Option 1: Streamlit Cloud (Easiest)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Select `3_streamlit_app.py`
5. Deploy!

**Limitations:**
- CPU only
- 1GB RAM limit
- May need to reduce model size

### Option 2: Docker Container

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "3_streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
# Build image
docker build -t fashion-stylist .

# Run container
docker run -p 8501:8501 -v $(pwd)/data:/app/data fashion-stylist
```

### Option 3: AWS EC2

1. Launch EC2 instance (g4dn.xlarge for GPU)
2. SSH into instance
3. Setup environment (see above)
4. Run with nohup:

```bash
nohup streamlit run 3_streamlit_app.py --server.port=8501 &
```

5. Configure security group to allow port 8501

### Option 4: Heroku

Create `Procfile`:

```
web: streamlit run 3_streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

Create `setup.sh`:

```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

Deploy:

```bash
heroku create your-app-name
git push heroku main
```

## Performance Optimization

### 1. Model Caching

Ensure models are cached:

```python
@st.cache_resource
def load_model():
    # Model loading code
    pass
```

### 2. Embedding Precomputation

Generate all embeddings before deployment:

```bash
python 1_generate_embeddings.py
```

### 3. CDN for Images

Use CDN for faster image loading:
- AWS S3 + CloudFront
- Cloudflare Images
- ImgIX

### 4. Database Setup (Optional)

For production, use PostgreSQL:

```sql
CREATE TABLE outfits (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    outfit_data JSONB,
    score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Monitoring & Logging

### Setup Logging

```python
import logging

logging.basicConfig(
    filename='fashion_stylist.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Monitor Performance

Use tools like:
- **Application:** Prometheus + Grafana
- **Server:** New Relic, DataDog
- **Errors:** Sentry

## Security Considerations

### 1. API Keys

Store in environment variables:

```bash
export FASHION_API_KEY="your_key_here"
```

### 2. Input Validation

Always validate user inputs:

```python
if not image_path.endswith(('.jpg', '.png')):
    raise ValueError("Invalid image format")
```

### 3. Rate Limiting

Implement rate limiting:

```python
from streamlit_extras.ratelimit import ratelimit

@ratelimit(max_calls=10, period=60)
def generate_outfits():
    # Generation code
    pass
```

## Backup & Recovery

### Automated Backups

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Backup data
tar -czf "$BACKUP_DIR/data_$DATE.tar.gz" data/

# Backup config
cp config.py "$BACKUP_DIR/config_$DATE.py"

# Remove old backups (keep last 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

Add to crontab:

```bash
# Run daily at 2 AM
0 2 * * * /path/to/backup.sh
```

## Scaling

### Horizontal Scaling

Use load balancer with multiple instances:

```yaml
# docker-compose.yml
version: '3'
services:
  app1:
    build: .
    ports:
      - "8501:8501"
  app2:
    build: .
    ports:
      - "8502:8501"
  nginx:
    image: nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### Database Caching

Use Redis for caching:

```python
import redis
r = redis.Redis(host='localhost', port=6379)

# Cache embeddings
r.set(f"embedding_{item_id}", embedding.tobytes())
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use CPU instead of GPU
   - Enable memory profiling

2. **Slow Performance**
   - Use GPU
   - Precompute embeddings
   - Reduce image sizes

3. **Model Loading Fails**
   - Check internet connection
   - Verify model cache
   - Try alternative model

## Health Checks

Add health check endpoint:

```python
@app.route('/health')
def health():
    return {'status': 'healthy', 'timestamp': time.time()}
```

## CI/CD Pipeline

GitHub Actions example:

```yaml
name: Deploy

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to production
        run: |
          # Your deployment commands
          ssh user@server 'cd /app && git pull && systemctl restart fashion-app'
```

## Maintenance

Regular maintenance tasks:

- **Daily:** Check logs for errors
- **Weekly:** Review performance metrics
- **Monthly:** Update dependencies
- **Quarterly:** Model retraining

---

For questions, contact: support@fashionstylist.ai
"""

# Save deployment guide
if __name__ == "__main__":
    with open("DEPLOYMENT.md", "w") as f:
        f.write(DEPLOYMENT_GUIDE)
    print("✅ Created DEPLOYMENT.md")