#!/usr/bin/env python3
"""
Utility functions for SWE Benchmark Patch Generator
Common utilities for logging, file operations, and data processing
"""

import os
import sys
import json
import logging
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import re


def setup_logging(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration with both file and console output
    
    Args:
        output_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("swe_benchmark")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler - detailed logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"swe_benchmark_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - simpler format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Error file handler - only errors
    error_file = os.path.join(logs_dir, f"errors_{timestamp}.log")
    error_handler = logging.FileHandler(error_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def create_output_dir(base_dir: str = "output") -> str:
    """
    Create timestamped output directory for results
    
    Args:
        base_dir: Base directory name for outputs
        
    Returns:
        Path to created output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "patches"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "contexts"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "analysis"), exist_ok=True)
    
    return output_dir


def safe_json_load(file_path: str, default: Any = None) -> Any:
    """
    Safely load JSON file with error handling
    
    Args:
        file_path: Path to JSON file
        default: Default value if file doesn't exist or can't be parsed
        
    Returns:
        Parsed JSON data or default value
    """
    try:
        if not os.path.exists(file_path):
            return default
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Failed to load JSON from {file_path}: {e}")
        return default


def safe_json_save(data: Any, file_path: str, indent: int = 2) -> bool:
    """
    Safely save data to JSON file with error handling
    
    Args:
        data: Data to save
        file_path: Path to save JSON file
        indent: JSON indentation level
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=str, ensure_ascii=False)
        return True
    except (TypeError, IOError) as e:
        logging.error(f"Failed to save JSON to {file_path}: {e}")
        return False


def calculate_file_hash(file_path: str, algorithm: str = "md5") -> Optional[str]:
    """
    Calculate hash of a file
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hash string or None if error
    """
    try:
        hash_obj = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except (IOError, ValueError) as e:
        logging.error(f"Failed to calculate hash for {file_path}: {e}")
        return None


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def clean_text(text: str, remove_extra_whitespace: bool = True) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Text to clean
        remove_extra_whitespace: Whether to remove extra whitespace
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove extra whitespace if requested
    if remove_extra_whitespace:
        # Remove trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        # Remove multiple consecutive empty lines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    return text.strip()


def extract_code_blocks(text: str, language: Optional[str] = None) -> List[str]:
    """
    Extract code blocks from markdown-formatted text
    
    Args:
        text: Text containing code blocks
        language: Specific language to extract (None for all)
        
    Returns:
        List of code block contents
    """
    if language:
        pattern = rf'```{re.escape(language)}\n(.*?)\n```'
    else:
        pattern = r'```(?:\w+)?\n(.*?)\n```'
    
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def count_tokens_estimate(text: str, model_type: str = "gpt") -> int:
    """
    Rough estimate of token count for different model types
    
    Args:
        text: Text to count tokens for
        model_type: Model type (gpt, claude, etc.)
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Rough estimates based on model type
    if model_type.lower() in ["gpt", "openai"]:
        # GPT models: roughly 4 characters per token
        return len(text) // 4
    elif model_type.lower() in ["claude", "anthropic"]:
        # Claude models: roughly 3.5 characters per token
        return int(len(text) / 3.5)
    else:
        # Default estimate
        return len(text) // 4


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def format_bytes(bytes_count: int) -> str:
    """
    Format byte count to human readable format
    
    Args:
        bytes_count: Number of bytes
        
    Returns:
        Formatted byte string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f}{unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f}PB"


def ensure_directory(path: str) -> str:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        path: Directory path
        
    Returns:
        The path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_project_root() -> str:
    """
    Get the project root directory
    
    Returns:
        Path to project root
    """
    current_file = Path(__file__).resolve()
    # Go up until we find a directory with common project files
    for parent in current_file.parents:
        if any((parent / marker).exists() for marker in ['.git', 'setup.py', 'pyproject.toml', 'requirements.txt']):
            return str(parent)
    
    # Fallback to current working directory
    return os.getcwd()


def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """
    Validate file path
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path_obj = Path(file_path)
        
        # Check if path is valid
        if not path_obj.is_absolute() and not path_obj.exists():
            # Try to resolve relative path
            path_obj = Path(get_project_root()) / path_obj
        
        if must_exist and not path_obj.exists():
            return False
            
        return True
    except (OSError, ValueError):
        return False


def merge_dicts(dict1: Dict, dict2: Dict, deep: bool = True) -> Dict:
    """
    Merge two dictionaries.

    Args:
        dict1: First dictionary.
        dict2: Second dictionary (values here take precedence).
        deep: Whether to merge nested dictionaries recursively.

    Returns:
        Merged dictionary.
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if deep and (
            key in result and isinstance(result[key], dict) and isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value, deep=True)
        else:
            result[key] = value
    return result


def create_temp_file(content: str, suffix: str = ".txt", prefix: str = "swe_") -> str:
    """
    Create temporary file with content
    
    Args:
        content: Content to write to file
        suffix: File suffix
        prefix: File prefix
        
    Returns:
        Path to temporary file
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, prefix=prefix, delete=False) as f:
        f.write(content)
        return f.name


def parse_error_message(error: Exception) -> Dict[str, Any]:
    """
    Parse error message and extract useful information
    
    Args:
        error: Exception object
        
    Returns:
        Dictionary with error information
    """
    return {
        'type': type(error).__name__,
        'message': str(error),
        'module': getattr(error, '__module__', None),
        'traceback': getattr(error, '__traceback__', None) is not None
    }


def retry_with_backoff(func, max_retries: int = 3, backoff_factor: float = 1.0, exceptions: Tuple = (Exception,)):
    """
    Decorator for retrying function calls with exponential backoff
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        backoff_factor: Backoff multiplier
        exceptions: Exceptions to catch and retry on
        
    Returns:
        Decorated function
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = backoff_factor * (2 ** attempt)
                    logging.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logging.error(f"All {max_retries + 1} attempts failed")
        
        raise last_exception
    
    return wrapper


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'memory_total': psutil.virtual_memory().total if 'psutil' in sys.modules else None,
        'cwd': os.getcwd(),
        'user': os.getenv('USER') or os.getenv('USERNAME'),
        'timestamp': datetime.now().isoformat()
    }


# Context manager for temporary directory changes
class TemporaryDirectory:
    """Context manager for temporary directory operations"""
    
    def __init__(self, path: str):
        self.path = path
        self.original_cwd = None
    
    def __enter__(self):
        self.original_cwd = os.getcwd()
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        os.chdir(self.path)
        return self.path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_cwd:
            os.chdir(self.original_cwd)


if __name__ == "__main__":
    # Test utilities
    print("Testing SWE Benchmark Utilities")
    
    # Test output directory creation
    output_dir = create_output_dir("test_output")
    print(f"Created output directory: {output_dir}")
    
    # Test logging setup
    logger = setup_logging(output_dir)
    logger.info("Test log message")
    logger.warning("Test warning message")
    
    # Test JSON operations
    test_data = {"test": "data", "timestamp": datetime.now()}
    json_file = os.path.join(output_dir, "test.json")
    if safe_json_save(test_data, json_file):
        loaded_data = safe_json_load(json_file)
        print(f"JSON test successful: {loaded_data}")
    
    # Test text processing
    sample_text = "  This is a test   \n\n\n  with extra whitespace  \n\n"
    cleaned = clean_text(sample_text)
    print(f"Cleaned text: '{cleaned}'")
    
    # Test token estimation
    tokens = count_tokens_estimate("This is a sample text for token counting")
    print(f"Estimated tokens: {tokens}")
    
    # Test system info
    info = get_system_info()
    print(f"System info: {info}")
    
    print("All tests completed!")