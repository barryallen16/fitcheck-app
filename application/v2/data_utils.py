# data_utils.py - Data validation, cleaning, and preprocessing utilities

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from PIL import Image
import pandas as pd
from collections import Counter

try:
    from config import (
        REQUIRED_FIELDS, VALID_CATEGORIES, VALID_STYLES,
        VALID_OCCASIONS, VALID_WEATHER, SUPPORTED_FORMATS
    )
except ImportError:
    # Fallback defaults
    REQUIRED_FIELDS = ['image_path', 'classification']
    VALID_CATEGORIES = []
    VALID_STYLES = []
    VALID_OCCASIONS = []
    VALID_WEATHER = []
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.webp']

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    item_index: Optional[int] = None
    
    def __str__(self):
        status = "✅ Valid" if self.is_valid else "❌ Invalid"
        msg = [status]
        if self.errors:
            msg.append(f"Errors: {', '.join(self.errors)}")
        if self.warnings:
            msg.append(f"Warnings: {', '.join(self.warnings)}")
        return " | ".join(msg)


class DataValidator:
    """Validate wardrobe data"""
    
    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: If True, warnings are treated as errors
        """
        self.strict_mode = strict_mode
    
    def validate_item(self, item: Dict, index: int = 0) -> ValidationResult:
        """Validate a single wardrobe item"""
        errors = []
        warnings = []
        
        # Check required fields
        for field in REQUIRED_FIELDS:
            if '.' in field:
                # Nested field
                parts = field.split('.')
                obj = item
                for part in parts:
                    if not isinstance(obj, dict) or part not in obj:
                        errors.append(f"Missing required field: {field}")
                        break
                    obj = obj[part]
            else:
                if field not in item:
                    errors.append(f"Missing required field: {field}")
        
        # Validate image path
        if 'image_path' in item:
            img_path = item['image_path']
            
            # Check if path exists
            if not Path(img_path).exists():
                errors.append(f"Image file not found: {img_path}")
            else:
                # Check file extension
                ext = Path(img_path).suffix.lower()
                if ext not in SUPPORTED_FORMATS:
                    warnings.append(f"Unsupported image format: {ext}")
                
                # Try to open image
                try:
                    with Image.open(img_path) as img:
                        # Check image size
                        if img.size[0] < 100 or img.size[1] < 100:
                            warnings.append(f"Image too small: {img.size}")
                except Exception as e:
                    errors.append(f"Cannot open image: {str(e)}")
        
        # Validate classification
        if 'classification' in item:
            classification = item['classification']
            
            # Check category
            if 'category' in classification:
                category = classification['category']
                if VALID_CATEGORIES and category not in VALID_CATEGORIES:
                    warnings.append(f"Unknown category: {category}")
            
            # Check style
            if 'style' in classification:
                style = classification['style']
                if VALID_STYLES and style not in VALID_STYLES:
                    warnings.append(f"Unknown style: {style}")
            
            # Check occasions
            if 'occasions' in classification:
                occasions = classification['occasions']
                if not isinstance(occasions, list):
                    warnings.append("'occasions' should be a list")
                elif VALID_OCCASIONS:
                    for occ in occasions:
                        if occ not in VALID_OCCASIONS:
                            warnings.append(f"Unknown occasion: {occ}")
            
            # Check color_primary
            if 'color_primary' in classification:
                color = classification['color_primary']
                if not color or color.strip() == '':
                    warnings.append("Empty color_primary")
        
        # Determine if valid
        is_valid = len(errors) == 0
        if self.strict_mode:
            is_valid = is_valid and len(warnings) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            item_index=index
        )
    
    def validate_jsonl(self, jsonl_path: str) -> Tuple[List[ValidationResult], Dict]:
        """
        Validate entire JSONL file
        
        Returns:
            (validation_results, statistics)
        """
        results = []
        valid_count = 0
        invalid_count = 0
        
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    result = self.validate_item(item, index=i)
                    results.append(result)
                    
                    if result.is_valid:
                        valid_count += 1
                    else:
                        invalid_count += 1
                        
                except json.JSONDecodeError as e:
                    results.append(ValidationResult(
                        is_valid=False,
                        errors=[f"JSON parsing error: {str(e)}"],
                        warnings=[],
                        item_index=i
                    ))
                    invalid_count += 1
        
        stats = {
            'total': len(results),
            'valid': valid_count,
            'invalid': invalid_count,
            'valid_percentage': (valid_count / len(results) * 100) if results else 0
        }
        
        return results, stats


class DataCleaner:
    """Clean and normalize wardrobe data"""
    
    @staticmethod
    def normalize_color(color: str) -> str:
        """Normalize color names"""
        color_map = {
            'grey': 'gray',
            'off-white': 'off_white',
            'off white': 'off_white',
            'dark blue': 'navy',
            'light blue': 'sky_blue',
            'deep maroon': 'maroon',
            'mustard yellow': 'mustard',
        }
        return color_map.get(color.lower(), color.lower())
    
    @staticmethod
    def normalize_category(category: str) -> str:
        """Normalize category names"""
        return category.lower().replace(' ', '_')
    
    @staticmethod
    def ensure_list(value) -> List:
        """Ensure value is a list"""
        if isinstance(value, list):
            return value
        elif isinstance(value, str):
            return [value]
        else:
            return []
    
    def clean_item(self, item: Dict) -> Dict:
        """Clean and normalize a single item"""
        cleaned = item.copy()
        
        if 'classification' in cleaned:
            classification = cleaned['classification'].copy()
            
            # Normalize color
            if 'color_primary' in classification:
                classification['color_primary'] = self.normalize_color(
                    classification['color_primary']
                )
            
            # Normalize category
            if 'category' in classification:
                classification['category'] = self.normalize_category(
                    classification['category']
                )
            
            # Ensure lists
            for field in ['occasions', 'weather', 'embellishments', 'color_secondary']:
                if field in classification:
                    classification[field] = self.ensure_list(classification[field])
            
            # Remove empty strings from lists
            for field in ['occasions', 'weather', 'embellishments']:
                if field in classification:
                    classification[field] = [
                        x for x in classification[field] 
                        if x and str(x).strip()
                    ]
            
            cleaned['classification'] = classification
        
        return cleaned
    
    def clean_jsonl(self, input_path: str, output_path: str) -> Dict:
        """
        Clean entire JSONL file
        
        Returns:
            Statistics about cleaning process
        """
        cleaned_items = []
        stats = {
            'total': 0,
            'cleaned': 0,
            'skipped': 0,
            'errors': 0
        }
        
        with open(input_path, 'r') as f:
            for line in f:
                stats['total'] += 1
                try:
                    item = json.loads(line.strip())
                    cleaned = self.clean_item(item)
                    cleaned_items.append(cleaned)
                    stats['cleaned'] += 1
                except Exception as e:
                    logger.error(f"Error cleaning item {stats['total']}: {e}")
                    stats['errors'] += 1
        
        # Write cleaned data
        with open(output_path, 'w') as f:
            for item in cleaned_items:
                f.write(json.dumps(item) + '\n')
        
        return stats


class DataAnalyzer:
    """Analyze wardrobe data statistics"""
    
    def __init__(self, jsonl_path: str):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    self.data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        stats = {
            'total_items': len(self.data),
            'categories': self._count_field('classification.category'),
            'colors': self._count_field('classification.color_primary'),
            'styles': self._count_field('classification.style'),
            'occasions': self._count_list_field('classification.occasions'),
            'materials': self._count_field('classification.material'),
            'formality': self._count_field('classification.formality'),
            'gender': self._count_field('classification.gender')
        }
        return stats
    
    def _count_field(self, field_path: str) -> Dict:
        """Count occurrences of a field"""
        values = []
        for item in self.data:
            obj = item
            for part in field_path.split('.'):
                if isinstance(obj, dict) and part in obj:
                    obj = obj[part]
                else:
                    obj = None
                    break
            if obj:
                values.append(str(obj))
        return dict(Counter(values))
    
    def _count_list_field(self, field_path: str) -> Dict:
        """Count occurrences in list fields"""
        values = []
        for item in self.data:
            obj = item
            for part in field_path.split('.'):
                if isinstance(obj, dict) and part in obj:
                    obj = obj[part]
                else:
                    obj = None
                    break
            if obj and isinstance(obj, list):
                values.extend([str(x) for x in obj])
        return dict(Counter(values))
    
    def generate_report(self) -> str:
        """Generate a text report"""
        stats = self.get_statistics()
        
        report = []
        report.append("="*80)
        report.append("WARDROBE DATA ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"\nTotal Items: {stats['total_items']}\n")
        
        # Categories
        report.append("\n📦 CATEGORIES:")
        for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
            report.append(f"  {cat}: {count}")
        
        # Colors
        report.append("\n🎨 TOP COLORS:")
        for color, count in sorted(stats['colors'].items(), key=lambda x: -x[1])[:10]:
            report.append(f"  {color}: {count}")
        
        # Styles
        report.append("\n💃 STYLES:")
        for style, count in sorted(stats['styles'].items(), key=lambda x: -x[1]):
            report.append(f"  {style}: {count}")
        
        # Occasions
        report.append("\n🎉 OCCASIONS:")
        for occ, count in sorted(stats['occasions'].items(), key=lambda x: -x[1]):
            report.append(f"  {occ}: {count}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def export_to_excel(self, output_path: str):
        """Export statistics to Excel"""
        try:
            import pandas as pd
            
            # Create dataframe from items
            rows = []
            for item in self.data:
                row = {
                    'filename': item.get('filename', ''),
                    'image_path': item.get('image_path', '')
                }
                
                if 'classification' in item:
                    classification = item['classification']
                    row.update({
                        'category': classification.get('category', ''),
                        'color_primary': classification.get('color_primary', ''),
                        'style': classification.get('style', ''),
                        'material': classification.get('material', ''),
                        'formality': classification.get('formality', ''),
                        'gender': classification.get('gender', ''),
                        'occasions': ', '.join(classification.get('occasions', [])),
                        'weather': ', '.join(classification.get('weather', []))
                    })
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_excel(output_path, index=False)
            logger.info(f"Exported to Excel: {output_path}")
            
        except ImportError:
            logger.error("pandas and openpyxl required for Excel export")


def validate_and_report(jsonl_path: str, output_report: Optional[str] = None):
    """Validate data and generate report"""
    print(f"\n{'='*80}")
    print(f"VALIDATING: {jsonl_path}")
    print(f"{'='*80}\n")
    
    # Validate
    validator = DataValidator(strict_mode=False)
    results, stats = validator.validate_jsonl(jsonl_path)
    
    # Print summary
    print(f"Total items: {stats['total']}")
    print(f"Valid: {stats['valid']} ({stats['valid_percentage']:.1f}%)")
    print(f"Invalid: {stats['invalid']}")
    
    # Print first few errors
    if stats['invalid'] > 0:
        print(f"\nFirst 5 validation issues:")
        error_count = 0
        for result in results:
            if not result.is_valid and error_count < 5:
                print(f"\nItem {result.item_index}:")
                for error in result.errors:
                    print(f"  ❌ {error}")
                for warning in result.warnings:
                    print(f"  ⚠️  {warning}")
                error_count += 1
    
    # Save report
    if output_report:
        with open(output_report, 'w') as f:
            f.write(f"Validation Report: {jsonl_path}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Total items: {stats['total']}\n")
            f.write(f"Valid: {stats['valid']}\n")
            f.write(f"Invalid: {stats['invalid']}\n\n")
            
            for result in results:
                if not result.is_valid:
                    f.write(f"\nItem {result.item_index}:\n")
                    for error in result.errors:
                        f.write(f"  ERROR: {error}\n")
                    for warning in result.warnings:
                        f.write(f"  WARNING: {warning}\n")
        
        print(f"\n✅ Report saved to: {output_report}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_utils.py <jsonl_file>")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    
    # Validate
    validate_and_report(jsonl_file, output_report=f"{jsonl_file}.validation_report.txt")
    
    # Analyze
    print(f"\n{'='*80}")
    print("ANALYZING DATA")
    print(f"{'='*80}")
    
    analyzer = DataAnalyzer(jsonl_file)
    print(analyzer.generate_report())