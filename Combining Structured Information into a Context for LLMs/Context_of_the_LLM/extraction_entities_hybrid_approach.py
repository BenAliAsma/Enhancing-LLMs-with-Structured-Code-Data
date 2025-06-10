from src.config import commit, date, version, repo_name, problem_stmt
from src.config import commit, date, version, repo_name, problem_stmt
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
from gliner import GLiNER
from matplotlib_venn import venn2, venn2_circles
from typing import List, Dict, Any, Set, Tuple, Optional, Protocol
import math
from nltk.corpus import stopwords
import nltk
import keyword
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

# Download required NLTK data
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')


class ExtractionStrategy(Enum):
    """Available extraction strategies"""
    UNION = "union"
    INTERSECTION = "intersection"
    WEIGHTED = "weighted"


@dataclass
class EntityConfig:
    """Configuration for entity extraction"""
    labels: List[str] = field(default_factory=lambda: [
        "function", "class", "path", "example", "constraint", "module", "condition"
    ])

    patterns: Dict[str, str] = field(default_factory=lambda: {
        "function": r'(?<!\.)\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?=\()',
        "class": r'\b(?:class\s+|new\s+|m\.|models\.)([A-Z][a-zA-Z0-9_]*)\b',
        "variable": r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?==(?!=))',
        "path": r'(?:from\s+|import\s+)([\w\.]+)(?:\s+import\s+[\w\., ]+)?|https://github\.com/[^\s\'"]+',
        "example": r'(>>>.*?)(?=\n(?!\.\.\.|>>>)|$)',
        "module": r'\b(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_\.]*)',
        "condition": r'\b(if|while|for|try|except|finally|assert)\b'
    })

    programming_stopwords: Set[str] = field(default_factory=lambda: {
        *keyword.kwlist,
        'array', 'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool',
        'True', 'False', 'none', 'print', 'len', 'range', 'type', 'isinstance',
        'hasattr', 'getattr', 'setattr', 'input', 'output', 'open', 'close',
        'code', 'function', 'def', 'method', 'class', 'variable', 'parameter', 'argument',
        'return', 'value', 'result', 'data', 'file', 'string', 'number', 'object',
        'instance', 'index', 'key', 'item', 'element', 'record', 'field', 'row', 'column',
        'line', 'block', 'loop', 'condition', 'statement', 'expression', 'operator',
        'module', 'package', 'library', 'framework', 'api', 'script', 'program',
        'runtime', 'memory', 'pointer', 'reference', 'buffer', 'stream',
        'read', 'write', 'execute', 'call', 'invoke', 'define', 'declare',
        'assign', 'initialize', 'compute', 'calculate', 'process',
        'store', 'load', 'save', 'convert', 'cast', 'main',
        'exception', 'error', 'debug', 'trace', 'log', 'warning',
        'consider', 'however', 'suddenly', 'following', 'expected', 'complex',
        'simple', 'basic', 'advanced', 'general', 'specific', 'example',
        'matrix', 'array', 'vector', 'tensor', 'shape', 'dim', 'axis',
        'train', 'test', 'fit', 'predict', 'evaluate', 'model', 'accuracy', 'score'
    })

    # Expanded and more flexible class patterns for validation, not direct extraction
    valid_class_patterns: List[str] = field(default_factory=lambda: [
        r'^[A-Z][a-zA-Z0-9_]*$',  # Basic CamelCase or PascalCase
        r'.*[Mm]odel$', r'.*[Mm]atrix$', r'.*[Cc]oordinate$',  # Suffixes
        r'^[A-Z][a-z]*\d+[A-Z]*$',  # Like Linear1D
        r'^[A-Z][a-z]+(?:[A-Z][a-z]*)*$'  # More robust CamelCase
    ])

    # Known class suffixes and prefixes that indicate a class
    class_indicators: Dict[str, List[str]] = field(default_factory=lambda: {
        'suffixes': [
            'Model', 'Matrix', 'Handler', 'Manager', 'Builder', 'Factory',
            'Parser', 'Validator', 'Exception', 'Error', 'Service', 'Engine',
            'Client', 'Controller', 'Connector', 'Adapter', 'Wrapper', 'Config',
            'Strategy', 'Interface', 'View', 'Request', 'Response', 'Task',
            'Worker', 'Pipeline', 'Job', 'Command', 'Runner', 'Node', 'Element'
        ],
        'prefixes': [
            'Base', 'Abstract', 'Default', 'Custom', 'Simple', 'Async',
            'Secure', 'Lazy', 'Fast', 'Smart', 'Auto', 'Mock',
            'Model', 'Matrix', 'Handler', 'Manager', 'Builder', 'Factory',
            'Parser', 'Validator', 'Exception', 'Error'
        ],
        'contains': [
            'Model', 'Matrix', 'Transform', 'Coordinate', 'Projection', 'Compound',
            'Token', 'Graph', 'Node', 'Tree', 'Cache', 'Stream', 'Buffer',
            'Session', 'Pool', 'Event', 'Channel', 'Wrapper', 'Context',
            'Embed', 'Vector', 'Layer', 'Decoder', 'Encoder', 'Extractor',
            'Batch', 'Log', 'Auth', 'Path', 'Request', 'Task'
        ]
    })

    valid_functions: Set[str] = field(default_factory=lambda: {
        'fit', 'transform', 'predict', 'plot', 'show',
        'reshape', 'flatten', 'transpose', 'inverse', 'solve', 'optimize',
        'minimize', 'maximize', 'integrate', 'differentiate', 'interpolate'
    })

    known_modules: Set[str] = field(default_factory=lambda: {
        # Data manipulation & analysis
        'numpy', 'pandas', 'scipy',

        # Machine Learning & Deep Learning
        'sklearn', 'scikit-learn', 'xgboost', 'lightgbm',
        'catboost', 'tensorflow', 'keras', 'torch', 'pytorch',
        'fastai', 'jax',

        # Visualization
        'matplotlib', 'seaborn', 'plotly', 'bokeh', 'altair',

        # Natural Language Processing
        'nltk', 'spacy', 'transformers', 'gensim',

        # Data storage / retrieval
        'sqlalchemy', 'pymongo', 'psycopg2', 'mysql.connector',

        # Web development
        'flask', 'django', 'fastapi', 'bottle', 'tornado',

        # Scraping / HTTP
        'requests', 'urllib', 'httpx', 'aiohttp', 'beautifulsoup4', 'bs4', 'scrapy',

        # Utilities & Misc
        'os', 'sys', 'time', 'datetime', 're', 'math', 'statistics', 'itertools',
        'functools', 'collections', 'subprocess', 'shutil', 'pathlib', 'logging', 'json',
        'csv', 'pickle', 'copy', 'typing', 'threading', 'multiprocessing', 'inspect',

        # Cloud / Deployment / ML Ops
        'mlflow', 'dvc', 'joblib', 'onnx', 'boto3',

        # DataFrames / Big Data
        'modin', 'dask', 'vaex', 'pyarrow',

        # Testing
        'pytest', 'unittest', 'nose', 'hypothesis',

        # Notebook / Interactive / Widgets
        'ipython', 'jupyter', 'ipywidgets', 'nbformat',

        # Image & Audio Processing
        'opencv', 'PIL', 'pillow', 'imageio', 'scikit-image', 'soundfile', 'librosa',

        # Scientific computing
        'sympy', 'numba', 'cython', 'pymc3', 'theano',

        # Graph / Network
        'networkx', 'graphviz', 'pyvis',

        # Others
        'openai', 'pydantic', 'tqdm', 'rich', 'dotenv', 'configparser', 'argparse'
    })


@dataclass
class Entity:
    """Represents an extracted entity"""
    text: str
    label: str
    start: int
    end: int
    source: str
    confidence: float = 0.0

    def __hash__(self):
        return hash((self.text.lower(), self.label))

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.text.lower() == other.text.lower() and self.label == other.label


class EntityValidator:
    """Handles validation logic for different entity types"""

    def __init__(self, config: EntityConfig):
        self.config = config
        self.stopwords = set(stopwords.words('english'))

    def clean_entity_text(self, text: str, label: str) -> str:
        """Clean entity text based on its type"""
        if label == "example":
            text = re.sub(r'^(>>>|\.\.\.)\s*', '', text)  # Combined >>> and ...
            text = text.strip()
        elif label == "path":
            # For paths, we want the full path, not just the first part after from/import
            # The regex for path in EntityConfig should ideally capture this.
            # This cleaning might be redundant if the regex is perfect.
            pass
        elif label == "function":
            text = text.strip()

        return text

    def is_valid_class_name(self, text: str) -> bool:
        """Improved class name validation with more comprehensive patterns"""
        if len(text) < 2:
            return False

        # Must start with uppercase letter
        if not text[0].isupper():
            return False

        # Disallow if it's a common English stopword or programming stopword
        text_lower = text.lower()
        if text_lower in self.stopwords or text_lower in self.config.programming_stopwords:
            return False

        # Check against known patterns
        for pattern in self.config.valid_class_patterns:
            if re.fullmatch(pattern, text):  # Use fullmatch for strictness
                return True

        # Check class indicators (suffixes, prefixes, contains)
        indicators = self.config.class_indicators

        # Check suffixes
        if any(text.endswith(suffix) for suffix in indicators['suffixes']):
            return True

        # Check prefixes
        if any(text.startswith(prefix) for prefix in indicators['prefixes']):
            return True

        # Check contains
        if any(indicator in text for indicator in indicators['contains']):
            return True

        # If it's a simple capitalized word that doesn't fit other criteria, reject
        if text.isalpha() and text == text.capitalize():  # e.g., "The", "If"
            return False

        return False

    def is_valid_function_name(self, text: str) -> bool:
        """Validate function names"""
        # A function name should generally be lowercase_with_underscores or camelCase (but usually lowercase_with_underscores in Python)
        # and be a valid identifier.
        if not text.isidentifier():
            return False

        text_lower = text.lower()
        if text_lower in self.stopwords or text_lower in self.config.programming_stopwords:
            return False

        # Specific known functions are always valid
        if text_lower in self.config.valid_functions:
            return True

        # Heuristic: functions often contain underscores or are camelCase but start lowercase
        if re.match(r'^[a-z_][a-zA-Z0-9_]*$', text):
            return True

        return False

    def is_valid_entity(self, text: str, label: str) -> bool:
        """Main validation method"""
        text_clean = text.strip()
        text_lower = text_clean.lower()

        # Basic validation
        if (len(text_clean) < 2 or
            text_lower in self.config.programming_stopwords or
            text_lower in self.stopwords or
            all(c in string.punctuation + string.whitespace for c in text_clean) or
            text_clean.isdigit()):  # Reject if it's just a number
            return False

        # Type-specific validation
        validation_map = {
            "function": self.is_valid_function_name,
            "class": self.is_valid_class_name,
            "variable": self._is_valid_variable,
            "path": self._is_valid_path,
            "example": self._is_valid_example,
            "module": self._is_valid_module,
        }

        validator = validation_map.get(label)
        if validator:
            return validator(text_clean)

        return True  # Default for constraint, condition

    def _is_valid_variable(self, text: str) -> bool:
        """Validate variable names"""
        return (text.isidentifier() and
                len(text) >= 2 and
                text.lower() not in self.stopwords and
                text.lower() not in self.config.programming_stopwords and
                not text[0].isupper())  # Variables typically don't start with uppercase

    def _is_valid_path(self, text: str) -> bool:
        """Validate module paths"""
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*$', text):
            return False
        return '.' in text or text.lower() in self.config.known_modules

    def _is_valid_example(self, text: str) -> bool:
        """Validate code examples"""
        # An example should contain some meaningful code, not just punctuation or very short strings.
        return (len(text) >= 5 and
                not all(c in string.punctuation + string.whitespace for c in text) and
                any(c.isalnum() for c in text))  # Must contain at least one alphanumeric char

    def _is_valid_module(self, text: str) -> bool:
        """Validate module names"""
        return text.lower() in self.config.known_modules


class EntityExtractor(ABC):
    """Abstract base class for entity extractors"""

    @abstractmethod
    def extract(self, text: str) -> List[Entity]:
        """Extract entities from text"""
        pass


class RegexExtractor(EntityExtractor):
    """Regex-based entity extractor with improved patterns"""

    def __init__(self, config: EntityConfig, validator: EntityValidator):
        self.config = config
        self.validator = validator

    def extract(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns"""
        entities = []

        # Process main text and code blocks
        # Prioritize matches within code blocks by processing them first
        code_blocks_matches = list(re.finditer(r'```(?:python)?(.*?)```', text, re.DOTALL | re.IGNORECASE))
        extracted_from_code_ranges = []

        for block_match in code_blocks_matches:
            code_text = block_match.group(1)
            block_start_offset = block_match.start(1)
            block_end_offset = block_match.end(1)
            extracted_from_code_ranges.append((block_start_offset, block_end_offset))

            block_entities = self._extract_from_text(code_text, block_start_offset)
            for entity in block_entities:
                entity.confidence = min(1.0, entity.confidence + 0.2)  # Higher confidence for code
            entities.extend(block_entities)

        # Now extract from the rest of the text, avoiding code blocks
        segments = []
        last_end = 0
        for block_start, block_end in extracted_from_code_ranges:
            if last_end < block_start:
                segments.append((text[last_end:block_start], last_end))
            last_end = block_end
        if last_end < len(text):
            segments.append((text[last_end:], last_end))

        for segment_text, segment_offset in segments:
            entities.extend(self._extract_from_text(segment_text, segment_offset))

        return entities

    def _extract_from_text(self, text: str, offset: int = 0) -> List[Entity]:
        """Extract entities from a text segment"""
        entities = []

        for label, pattern in self.config.patterns.items():
            for match in re.finditer(pattern, text, re.MULTILINE | re.DOTALL):  # Removed IGNORECASE unless truly needed
                entity_text = self._extract_match_text(match, label)
                cleaned_text = self.validator.clean_entity_text(entity_text, label)

                # Check for empty or solely punctuation/whitespace after cleaning
                if not cleaned_text or all(c in string.punctuation + string.whitespace for c in cleaned_text):
                    continue

                if self.validator.is_valid_entity(cleaned_text, label):
                    entities.append(Entity(
                        text=cleaned_text,
                        label=label,
                        start=offset + match.start(),
                        end=offset + match.end(),
                        source="regex",
                        confidence=0.7  # Base confidence for regex
                    ))

        return entities

    def _extract_match_text(self, match: re.Match, label: str) -> str:
        """Extract text from regex match based on label type"""
        # For 'class' and 'function', group(1) should contain the name.
        # For 'example', group(0) is the whole match.
        # For 'path', group(1) is the full path.
        if label in ["function", "class", "path", "variable"]:
            return match.group(1) if match.groups() else match.group(0)
        elif label == "example":
            return match.group(0).strip()
        else:  # For constraint, module, condition - often just group(0)
            return match.group(0)


class GLiNERExtractor(EntityExtractor):
    """GLiNER-based entity extractor"""

    def __init__(self, config: EntityConfig, validator: EntityValidator, model_name: str = "urchade/gliner_medium-v2.1"):
        self.config = config
        self.validator = validator
        self.model = GLiNER.from_pretrained(model_name)

    def extract(self, text: str) -> List[Entity]:
        """Extract entities using GLiNER"""
        try:
            gliner_ents = self.model.predict_entities(text, self.config.labels, threshold=0.3)
        except Exception as e:
            print(f"GLiNER extraction failed: {e}")
            return []

        entities = []
        for ent in gliner_ents:
            cleaned_text = self.validator.clean_entity_text(ent["text"], ent["label"])
            # Ensure text is not empty or just punctuation after cleaning
            if not cleaned_text or all(c in string.punctuation + string.whitespace for c in cleaned_text):
                continue

            if cleaned_text and self.validator.is_valid_entity(cleaned_text, ent["label"]):
                entities.append(Entity(
                    text=cleaned_text,
                    label=ent["label"],
                    start=ent.get("start", 0),
                    end=ent.get("end", 0),
                    source="gliner",
                    confidence=ent.get("score", 0.5)
                ))

        return entities


class EntityCombiner:
    """Handles combining entities from multiple extractors"""

    @staticmethod
    def combine(regex_entities: List[Entity], gliner_entities: List[Entity],
                strategy: ExtractionStrategy) -> List[Entity]:
        """Combine entities using specified strategy"""
        if strategy == ExtractionStrategy.INTERSECTION:
            return EntityCombiner._intersection_strategy(regex_entities, gliner_entities)
        elif strategy == ExtractionStrategy.WEIGHTED:
            return EntityCombiner._weighted_strategy(regex_entities, gliner_entities)
        else:  # UNION
            return EntityCombiner._union_strategy(regex_entities, gliner_entities)

    @staticmethod
    def _intersection_strategy(regex_entities: List[Entity], gliner_entities: List[Entity]) -> List[Entity]:
        """Only entities found by both methods"""
        regex_set = set(regex_entities)
        return [e for e in gliner_entities if e in regex_set]

    @staticmethod
    def _weighted_strategy(regex_entities: List[Entity], gliner_entities: List[Entity]) -> List[Entity]:
        """Combine with confidence weighting"""
        combined_entities = {}

        # Add GLiNER entities first, they tend to be more contextually accurate
        for entity in gliner_entities:
            key = (entity.text.lower(), entity.label)
            combined_entities[key] = entity

        # Add regex entities. If a key already exists (from GLiNER), boost confidence.
        for entity in regex_entities:
            key = (entity.text.lower(), entity.label)
            if key in combined_entities:
                # Entity found by both, increase confidence of the GLiNER version
                combined_entities[key].confidence = min(1.0, combined_entities[key].confidence * 1.5)
                combined_entities[key].source = "both"
            else:
                combined_entities[key] = entity

        return list(combined_entities.values())

    @staticmethod
    def _union_strategy(regex_entities: List[Entity], gliner_entities: List[Entity]) -> List[Entity]:
        """Combine all entities, removing duplicates"""
        combined = []
        seen_keys = set()

        # Add GLiNER entities, prioritizing them for duplicates
        for entity in gliner_entities:
            key = (entity.text.lower(), entity.label)
            if key not in seen_keys:
                seen_keys.add(key)
                combined.append(entity)

        # Add regex entities if not already seen
        for entity in regex_entities:
            key = (entity.text.lower(), entity.label)
            if key not in seen_keys:
                seen_keys.add(key)
                combined.append(entity)

        return combined


class MetricsCalculator:
    """Calculates extraction metrics"""

    @staticmethod
    def calculate(regex_entities: List[Entity], gliner_entities: List[Entity]) -> Dict[str, Any]:
        """Calculate comprehensive metrics"""
        regex_set = set(regex_entities)
        gliner_set = set(gliner_entities)

        intersection = regex_set & gliner_set
        union = regex_set | gliner_set

        return {
            "counts": {
                "regex_only": len(regex_set - gliner_set),
                "gliner_only": len(gliner_set - regex_set),
                "both_methods": len(intersection),
                "total_unique": len(union)
            },
            "similarity_metrics": {
                "jaccard_index": len(intersection) / len(union) if union else 0,
                "overlap_coefficient": len(intersection) / min(len(regex_set), len(gliner_set)) if min(len(regex_set), len(gliner_set)) > 0 else 0,
                "dice_coefficient": 2 * len(intersection) / (len(regex_set) + len(gliner_set)) if (len(regex_set) + len(gliner_set)) > 0 else 0,
            },
            "coverage_metrics": {
                "regex_coverage_of_gliner": len(intersection) / len(gliner_set) if gliner_set else 0,
                "gliner_coverage_of_regex": len(intersection) / len(regex_set) if regex_set else 0
            },
            "type_distribution": {
                "regex": dict(Counter(e.label for e in regex_entities)),
                "gliner": dict(Counter(e.label for e in gliner_entities))
            }
        }


class Visualizer:
    """Handles visualization of extraction results"""

    @staticmethod
    def create_visualizations(entities: List[Entity], regex_entities: List[Entity],
                              gliner_entities: List[Entity], metrics: Dict[str, Any],
                              output_prefix: str = "improved_hybrid"):
        """Create comprehensive visualizations"""
        if not entities:
            print("No entities to visualize")
            return

        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Entity distribution plot
        Visualizer._create_distribution_plot(entities, regex_entities, gliner_entities, output_prefix)

        # 2. Metrics dashboard
        Visualizer._create_metrics_dashboard(entities, regex_entities, gliner_entities, metrics, output_prefix)

        return {
            "total_entities": len(entities),
            "metrics": metrics,
            "visualizations_saved": [f"{output_prefix}_distribution.png", f"{output_prefix}_analysis.png"]
        }

    @staticmethod
    def _create_distribution_plot(entities, regex_entities, gliner_entities, output_prefix):
        """Create entity distribution visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Count by type and method
        regex_types = Counter(e.label for e in regex_entities)
        gliner_types = Counter(e.label for e in gliner_entities)
        hybrid_types = Counter(e.label for e in entities)

        all_types = sorted(set(regex_types.keys()) | set(gliner_types.keys()) | set(hybrid_types.keys()))

        # Stacked bar chart
        x = np.arange(len(all_types))
        width = 0.6

        regex_counts = [regex_types.get(t, 0) for t in all_types]
        gliner_counts = [gliner_types.get(t, 0) for t in all_types]

        ax1.bar(x, regex_counts, width, label='Regex Only', alpha=0.8)
        ax1.bar(x, gliner_counts, width, bottom=regex_counts, label='GLiNER', alpha=0.8)

        ax1.set_xlabel('Entity Type')
        ax1.set_ylabel('Count')
        ax1.set_title('Entity Extraction by Method and Type')
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_types, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Pie chart for hybrid results
        hybrid_counts = [hybrid_types.get(t, 0) for t in all_types]
        if hybrid_counts and sum(hybrid_counts) > 0:
            ax2.pie(hybrid_counts, labels=all_types, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Hybrid Extraction Distribution')

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def _create_metrics_dashboard(entities, regex_entities, gliner_entities, metrics, output_prefix):
        """Create metrics dashboard"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Similarity metrics
        sim_metrics = metrics['similarity_metrics']
        metric_names = list(sim_metrics.keys())
        metric_values = list(sim_metrics.values())

        bars1 = ax1.barh(metric_names, metric_values, color='skyblue')
        ax1.set_xlim(0, 1)
        ax1.set_xlabel('Score')
        ax1.set_title('Similarity Metrics')
        ax1.grid(axis='x', alpha=0.3)

        for i, v in enumerate(metric_values):
            ax1.text(v + 0.02, i, f'{v:.3f}', va='center')

        # Coverage metrics
        cov_metrics = metrics['coverage_metrics']
        cov_names = ['Regex→GLiNER', 'GLiNER→Regex']
        cov_values = list(cov_metrics.values())

        ax2.bar(cov_names, cov_values, color=['lightcoral', 'lightgreen'])
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Coverage Score')
        ax2.set_title('Coverage Analysis')
        ax2.grid(axis='y', alpha=0.3)

        for i, v in enumerate(cov_values):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center')

        # Method comparison
        method_counts = [
            metrics['counts']['regex_only'],
            metrics['counts']['gliner_only'],
            metrics['counts']['both_methods']
        ]
        method_labels = ['Regex Only', 'GLiNER Only', 'Both Methods']

        ax3.pie(method_counts, labels=method_labels, autopct='%1.1f%%',
                colors=['lightblue', 'lightgreen', 'orange'])
        ax3.set_title('Entity Source Distribution')

        # Summary stats
        stats_text = f"""
        Total Unique Entities: {metrics['counts']['total_unique']}

        Method Performance:
        • Regex: {len(regex_entities)} entities
        • GLiNER: {len(gliner_entities)} entities
        • Hybrid: {len(entities)} entities

        Quality Metrics:
        • Jaccard Index: {sim_metrics['jaccard_index']:.3f}
        • Dice Coefficient: {sim_metrics['dice_coefficient']:.3f}
        """

        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Summary Statistics')

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


class HybridEntityExtractor:
    """Main orchestrator class for hybrid entity extraction"""

    def __init__(self, config: Optional[EntityConfig] = None):
        self.config = config or EntityConfig()
        self.validator = EntityValidator(self.config)
        self.regex_extractor = RegexExtractor(self.config, self.validator)
        self.gliner_extractor = GLiNERExtractor(self.config, self.validator)

        # Storage for analysis
        self.last_regex_entities: List[Entity] = []
        self.last_gliner_entities: List[Entity] = []

    def extract(self, text: str, strategy: ExtractionStrategy = ExtractionStrategy.UNION) -> List[Entity]:
        """Extract entities using hybrid approach"""
        # Extract using both methods
        self.last_regex_entities = self.regex_extractor.extract(text)
        self.last_gliner_entities = self.gliner_extractor.extract(text)

        # Combine results
        return EntityCombiner.combine(
            self.last_regex_entities,
            self.last_gliner_entities,
            strategy
        )

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for the last extraction"""
        return MetricsCalculator.calculate(self.last_regex_entities, self.last_gliner_entities)

    def visualize(self, entities: List[Entity], output_prefix: str = "improved_hybrid") -> Dict[str, Any]:
        """Create visualizations for extraction results"""
        metrics = self.calculate_metrics()
        return Visualizer.create_visualizations(
            entities, self.last_regex_entities, self.last_gliner_entities,
            metrics, output_prefix
        )


# Example usage and testing
def main():

    # Create extractor with default configuration
    extractor = HybridEntityExtractor()

    print("Testing improved modular entity extraction...")
    print("=" * 60)

    # Extract entities
    entities= extractor.extract(problem_stmt, ExtractionStrategy.UNION)

    # Display results
    print(f"{'Entity':<25} | {'Type':<10} | {'Source':<8}")
    print("-" * 70)

    for entity in sorted(entities, key=lambda x: (x.label, x.text)):
        entity_text = entity.text[:24]
        confidence = f"{entity.confidence:.2f}"
        print(f"{entity_text:<25} | {entity.label:<10} | {entity.source:<8} ")

    # Calculate and display metrics
    metrics = extractor.calculate_metrics()
    print(f"\n Summary:")
    print(f"Total entities extracted: {len(entities)}")
    print(f"Regex entities: {len(extractor.last_regex_entities)}")
    print(f"GLiNER entities: {len(extractor.last_gliner_entities)}")
    print(f"Agreement: {metrics['counts']['both_methods']} entities")
    print(f"Jaccard Index: {metrics['similarity_metrics']['jaccard_index']:.3f}")

    # Create visualizations
    results = extractor.visualize(entities)
    print(f"Visualizations saved: {results['visualizations_saved']}")
    return entities, extractor

if __name__ == "__main__":
    entities, extractor = main()