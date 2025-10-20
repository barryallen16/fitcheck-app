# export_outfits.py - Export outfit recommendations to various formats

import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from PIL import Image
import io

try:
    from outfit_generator import OutfitGenerator
except ImportError:
    print("Error: outfit_generator.py not found")
    exit(1)


class OutfitExporter:
    """Export outfits to various formats"""
    
    def __init__(self, generator: OutfitGenerator):
        self.generator = generator
    
    def export_to_json(self, outfits: List[Dict], output_path: str):
        """Export outfits to JSON"""
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'num_outfits': len(outfits),
            'outfits': []
        }
        
        for i, outfit in enumerate(outfits, 1):
            outfit_data = {
                'outfit_id': i,
                'type': outfit['type'],
                'structure': outfit['structure'],
                'scores': outfit['score_dict'],
                'items': []
            }
            
            for item in outfit['items']:
                classification = item.get('classification', {})
                outfit_data['items'].append({
                    'filename': item.get('filename', ''),
                    'image_path': item.get('image_path', ''),
                    'category': classification.get('category', ''),
                    'color': classification.get('color_primary', ''),
                    'style': classification.get('style', ''),
                    'material': classification.get('material', ''),
                    'specific_type': classification.get('specific_type', '')
                })
            
            export_data['outfits'].append(outfit_data)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Exported to JSON: {output_path}")
    
    def export_to_html(self, outfits: List[Dict], output_path: str, include_images: bool = True):
        """Export outfits to HTML report"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Fashion Recommendations</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .outfit-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .outfit-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .outfit-title {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }
        .score-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
        }
        .scores {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }
        .score-item {
            flex: 1;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        .score-label {
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
        }
        .score-value {
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }
        .items-container {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .item-card {
            flex: 1;
            min-width: 200px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            background: #fafafa;
        }
        .item-image {
            width: 100%;
            height: 250px;
            object-fit: cover;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .item-details {
            font-size: 0.9em;
        }
        .item-category {
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }
        .item-attribute {
            margin: 3px 0;
            color: #666;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            color: #999;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>👗 AI Fashion Recommendations</h1>
    <p style="text-align: center; color: #666;">Generated on """ + datetime.now().strftime("%B %d, %Y at %I:%M %p") + """</p>
"""
        
        for i, outfit in enumerate(outfits, 1):
            score = outfit['score']
            
            html += f"""
    <div class="outfit-card">
        <div class="outfit-header">
            <div class="outfit-title">Outfit #{i}: {outfit['type'].replace('_', ' ').title()}</div>
            <div class="score-badge">{score.overall:.0%} Match</div>
        </div>
        
        <div class="scores">
            <div class="score-item">
                <div class="score-label">Visual Harmony</div>
                <div class="score-value">{score.visual_coherence:.0%}</div>
            </div>
            <div class="score-item">
                <div class="score-label">Color Harmony</div>
                <div class="score-value">{score.color_harmony:.0%}</div>
            </div>
            <div class="score-item">
                <div class="score-label">Style Match</div>
                <div class="score-value">{score.style_compatibility:.0%}</div>
            </div>
            <div class="score-item">
                <div class="score-label">Occasion Fit</div>
                <div class="score-value">{score.occasion_fit:.0%}</div>
            </div>
        </div>
        
        <div class="items-container">
"""
            
            for item in outfit['items']:
                classification = item.get('classification', {})
                img_path = item.get('image_path', '')
                
                html += f"""
            <div class="item-card">
"""
                if include_images and Path(img_path).exists():
                    html += f"""
                <img src="file://{Path(img_path).absolute()}" class="item-image" alt="{classification.get('category', 'Item')}">
"""
                
                html += f"""
                <div class="item-details">
                    <div class="item-category">{classification.get('category', 'Unknown').replace('_', ' ').title()}</div>
                    <div class="item-attribute">🎨 {classification.get('color_primary', 'N/A')}</div>
                    <div class="item-attribute">✨ {classification.get('style', 'N/A')}</div>
                    <div class="item-attribute">📦 {classification.get('material', 'N/A')}</div>
                    <div class="item-attribute">👔 {classification.get('specific_type', 'N/A')}</div>
                </div>
            </div>
"""
            
            html += """
        </div>
    </div>
"""
        
        html += """
    <div class="footer">
        <p>Generated by AI Fashion Stylist • Powered by Marqo-FashionSigLIP</p>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ Exported to HTML: {output_path}")
    
    def export_to_markdown(self, outfits: List[Dict], output_path: str):
        """Export outfits to Markdown"""
        md = f"""# 👗 Fashion Recommendations

**Generated:** {datetime.now().strftime("%B %d, %Y at %I:%M %p")}  
**Total Outfits:** {len(outfits)}

---

"""
        
        for i, outfit in enumerate(outfits, 1):
            score = outfit['score']
            
            md += f"""## Outfit #{i}: {outfit['type'].replace('_', ' ').title()}

**Overall Score:** {score.overall:.0%} • 
Visual: {score.visual_coherence:.0%} • 
Color: {score.color_harmony:.0%} • 
Style: {score.style_compatibility:.0%} • 
Occasion: {score.occasion_fit:.0%}

### Items:

"""
            
            for item in outfit['items']:
                classification = item.get('classification', {})
                md += f"""- **{classification.get('category', 'Item').replace('_', ' ').title()}**
  - Color: {classification.get('color_primary', 'N/A')}
  - Style: {classification.get('style', 'N/A')}
  - Material: {classification.get('material', 'N/A')}
  - Type: {classification.get('specific_type', 'N/A')}

"""
            
            md += "---\n\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md)
        
        print(f"✅ Exported to Markdown: {output_path}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export outfit recommendations')
    parser.add_argument('--wardrobe', default='men', choices=['men', 'women'],
                       help='Wardrobe type')
    parser.add_argument('--occasion', default='casual', help='Occasion filter')
    parser.add_argument('--style', default='any', help='Style filter')
    parser.add_argument('--format', default='html', choices=['json', 'html', 'markdown'],
                       help='Export format')
    parser.add_argument('--output', default='exports/recommendations',
                       help='Output file prefix')
    parser.add_argument('--num', type=int, default=5, help='Number of outfits')
    
    args = parser.parse_args()
    
    # Initialize generator
    print("Loading wardrobe...")
    generator = OutfitGenerator(
        metadata_jsonl=f"data/{args.wardrobe}_wardrobe_with_embeddings.jsonl",
        embeddings_npy=f"data/{args.wardrobe}_wardrobe_embeddings.npy"
    )
    
    # Generate recommendations
    print(f"Generating {args.num} outfits...")
    outfits = generator.recommend_outfits(
        occasion=args.occasion,
        style=args.style,
        num_outfits=args.num
    )
    
    if not outfits:
        print("❌ No outfits found")
        return
    
    # Export
    Path('exports').mkdir(exist_ok=True)
    exporter = OutfitExporter(generator)
    
    if args.format == 'json':
        exporter.export_to_json(outfits, f"{args.output}.json")
    elif args.format == 'html':
        exporter.export_to_html(outfits, f"{args.output}.html")
    elif args.format == 'markdown':
        exporter.export_to_markdown(outfits, f"{args.output}.md")
    
    print(f"\n✅ Exported {len(outfits)} outfits")


if __name__ == "__main__":
    main()


# ============================================================================
# batch_process.py - Batch processing utilities
# ============================================================================

"""
Batch processing script for large wardrobes
Processes data in chunks to avoid memory issues
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional


class BatchProcessor:
    """Process wardrobe data in batches"""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
    
    def process_in_batches(
        self,
        input_jsonl: str,
        output_dir: str,
        start_index: int = 0,
        end_index: Optional[int] = None,
        processor_func=None
    ) -> Dict:
        """
        Process JSONL file in batches
        
        Args:
            input_jsonl: Input file path
            output_dir: Directory for batch outputs
            start_index: Starting line number
            end_index: Ending line number (None for all)
            processor_func: Function to process each item
        
        Returns:
            Statistics dictionary
        """
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        stats = {
            'total_read': 0,
            'total_processed': 0,
            'batches': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        current_batch = []
        batch_num = 0
        
        with open(input_jsonl, 'r') as f:
            for line_num, line in enumerate(f):
                # Skip until start_index
                if line_num < start_index:
                    continue
                
                # Stop at end_index
                if end_index and line_num >= end_index:
                    break
                
                stats['total_read'] += 1
                
                try:
                    item = json.loads(line.strip())
                    
                    # Process item if function provided
                    if processor_func:
                        item = processor_func(item)
                    
                    current_batch.append(item)
                    stats['total_processed'] += 1
                    
                    # Save batch when full
                    if len(current_batch) >= self.batch_size:
                        batch_num += 1
                        self._save_batch(current_batch, output_dir, batch_num)
                        stats['batches'] += 1
                        current_batch = []
                        
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    stats['errors'] += 1
        
        # Save final batch
        if current_batch:
            batch_num += 1
            self._save_batch(current_batch, output_dir, batch_num)
            stats['batches'] += 1
        
        stats['elapsed_time'] = time.time() - stats['start_time']
        
        return stats
    
    def _save_batch(self, batch: List[Dict], output_dir: str, batch_num: int):
        """Save a batch to file"""
        output_path = Path(output_dir) / f"batch_{batch_num:04d}.jsonl"
        with open(output_path, 'w') as f:
            for item in batch:
                f.write(json.dumps(item) + '\n')
        print(f"✅ Saved batch {batch_num}: {len(batch)} items")
    
    def merge_batches(self, batch_dir: str, output_file: str):
        """Merge all batch files into single output"""
        batch_files = sorted(Path(batch_dir).glob("batch_*.jsonl"))
        
        total_items = 0
        with open(output_file, 'w') as out_f:
            for batch_file in batch_files:
                with open(batch_file, 'r') as in_f:
                    for line in in_f:
                        out_f.write(line)
                        total_items += 1
        
        print(f"✅ Merged {len(batch_files)} batches into {output_file}")
        print(f"✅ Total items: {total_items}")


def process_large_wardrobe_example():
    """Example: Process large wardrobe in batches"""
    
    processor = BatchProcessor(batch_size=50)
    
    # Process in batches
    stats = processor.process_in_batches(
        input_jsonl="data/large_wardrobe.jsonl",
        output_dir="data/batches",
        start_index=0,
        end_index=1000  # Process first 1000 items
    )
    
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    print(f"Total read: {stats['total_read']}")
    print(f"Total processed: {stats['total_processed']}")
    print(f"Batches created: {stats['batches']}")
    print(f"Errors: {stats['errors']}")
    print(f"Time: {stats['elapsed_time']:.2f} seconds")
    
    # Merge batches
    processor.merge_batches(
        batch_dir="data/batches",
        output_file="data/processed_wardrobe.jsonl"
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        process_large_wardrobe_example()
    else:
        main()