import os
import json
import logging
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

def setup_logging(log_dir: str = "logs", level: int = logging.INFO):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"swe_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def validate_patch_format(patch: str) -> Dict[str, Any]:
    """Validate if patch follows proper git diff format"""
    result = {
        'is_valid': False,
        'has_diff_header': False,
        'has_file_paths': False,
        'has_line_numbers': False,
        'has_changes': False,
        'warnings': []
    }
    
    # Check for diff header
    if re.search(r'diff --git', patch):
        result['has_diff_header'] = True
    else:
        result['warnings'].append('Missing diff header')
    
    # Check for file paths
    if re.search(r'--- a/.*\n\+\+\+ b/', patch):
        result['has_file_paths'] = True
    else:
        result['warnings'].append('Missing file path indicators')
    
    # Check for line numbers
    if re.search(r'@@.*@@', patch):
        result['has_line_numbers'] = True
    else:
        result['warnings'].append('Missing line number indicators')
    
    # Check for actual changes
    if re.search(r'^[+-]', patch, re.MULTILINE):
        result['has_changes'] = True
    else:
        result['warnings'].append('No actual changes detected')
    
    result['is_valid'] = all([
        result['has_diff_header'],
        result['has_file_paths'],
        result['has_line_numbers'],
        result['has_changes']
    ])
    
    return result

def extract_code_from_patch(patch: str) -> Dict[str, List[str]]:
    """Extract added and removed code from patch"""
    lines = patch.split('\n')
    
    added_lines = []
    removed_lines = []
    
    for line in lines:
        if line.startswith('+') and not line.startswith('+++'):
            added_lines.append(line[1:])  # Remove + prefix
        elif line.startswith('-') and not line.startswith('---'):
            removed_lines.append(line[1:])  # Remove - prefix
    
    return {
        'added': added_lines,
        'removed': removed_lines
    }

def calculate_patch_hash(patch: str) -> str:
    """Calculate hash of patch for deduplication"""
    return hashlib.md5(patch.encode()).hexdigest()

def save_json(data: Any, filepath: str, indent: int = 2):
    """Save data to JSON file with error handling"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except Exception as e:
        logging.error(f"Failed to save JSON to {filepath}: {str(e)}")
        return False

def load_json(filepath: str) -> Optional[Dict]:
    """Load JSON file with error handling"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {filepath}: {str(e)}")
        return None

def create_directory_structure(base_dir: str):
    """Create the complete directory structure for the project"""
    dirs = [
        "src",
        "data",
        "data/outputs",
        "results",
        "logs"
    ]
    
    for dir_name in dirs:
        full_path = os.path.join(base_dir, dir_name)
        os.makedirs(full_path, exist_ok=True)
    
    return True

def benchmark_function(func, *args, **kwargs):
    """Benchmark a function execution"""
    start_time = datetime.now()
    result = func(*args, **kwargs)
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    return result, execution_time