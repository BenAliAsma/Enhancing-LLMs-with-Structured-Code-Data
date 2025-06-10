import re
from src.config import commit, date, version, repo_name, problem_stmt
import numpy as np
from dataclasses import dataclass
from typing import List
from extraction_entities_hybrid_approach import main as extract_main
from operator import attrgetter

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    source: str
    confidence: float
    # Add the new fields that were causing the error
    bm25_score: float = 0.0
    in_code: bool = False
    near_error: bool = False

def generate_query_from_problem(problem_stmt: str) -> str:
    """
    Automatically generate a query from the problem statement.
    
    Args:
        problem_stmt: The problem statement text
        
    Returns:
        Generated query string
    """
    import re
    from collections import Counter
    
    # Technical domain keywords to prioritize
    technical_keywords = {
        'matrix', 'model', 'modeling', 'algorithm', 'function', 'class', 'method',
        'implementation', 'optimization', 'learning', 'prediction', 'classification',
        'regression', 'neural', 'network', 'deep', 'machine', 'data', 'analysis',
        'computation', 'calculation', 'processing', 'transformation', 'feature',
        'parameter', 'variable', 'dimension', 'vector', 'tensor', 'array',
        'separability', 'clustering', 'embedding', 'encoding', 'decoding',
        'training', 'testing', 'validation', 'accuracy', 'performance'
    }
    
    # Action/task keywords
    action_keywords = {
        'implement', 'create', 'build', 'develop', 'design', 'construct',
        'fix', 'solve', 'optimize', 'improve', 'enhance', 'modify',
        'calculate', 'compute', 'process', 'transform', 'convert',
        'train', 'predict', 'classify', 'analyze', 'evaluate'
    }
    
    # Clean the problem statement
    text = re.sub(r'```.*?```', '', problem_stmt, flags=re.DOTALL)  # Remove code blocks
    text = re.sub(r'[^\w\s]', ' ', text.lower())  # Remove punctuation, lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    
    # Extract words
    words = text.split()
    
    # Filter words (remove common stopwords but keep technical terms)
    common_stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
        'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
        'whom', 'whose', 'where', 'when', 'why', 'how', 'all', 'any', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
        'will', 'just', 'should', 'now', 'get', 'use', 'using', 'used'
    }
    
    filtered_words = [w for w in words if w not in common_stopwords and len(w) > 2]
    
    # Count word frequencies
    word_counts = Counter(filtered_words)
    
    # Score words based on importance
    scored_words = {}
    for word, count in word_counts.items():
        score = count  # Base frequency score
        
        # Boost technical keywords
        if word in technical_keywords:
            score *= 3
        
        # Boost action keywords
        if word in action_keywords:
            score *= 2
            
        # Boost words that appear in uppercase in original text (likely important)
        if word.upper() in problem_stmt:
            score *= 1.5
            
        scored_words[word] = score
    
    # Select top words for query
    top_words = sorted(scored_words.items(), key=lambda x: x[1], reverse=True)[:6]
    
    # Extract just the words
    query_words = [word for word, score in top_words]
    
    # Join into query string
    query = ' '.join(query_words)
    
    return query if query else "modeling implementation"  # Fallback

def extract_code_and_error_positions(problem_stmt):
    code_positions = []
    error_positions = []

    # Extract code blocks between triple backticks
    for match in re.finditer(r"```python(.*?)```", problem_stmt, re.DOTALL):
        start, end = match.span()
        code_positions.append((start, end))

    # Look for error-indicating lines
    error_patterns = [
        r'\bdoes not\b',
        r'\bnot\b',
        r'\berror\b',
        r'\bmissing\b',
        r'\bunexpected\b',
        r'\bbug\b',
        r'\bissue\b',
        r'\bproblem\b',
        r'\bincorrect\b',
        r'\bwrong\b'
    ]
    error_regex = re.compile('|'.join(error_patterns), re.IGNORECASE)

    for match in error_regex.finditer(problem_stmt):
        # Capture context window around the match
        start = max(0, match.start() - 40)
        end = min(len(problem_stmt), match.end() + 40)
        error_positions.append((start, end))

    return code_positions, error_positions

def calculate_entity_scores(entities: List[Entity], weights: dict,
                          code_positions: List[tuple] = None,
                          error_positions: List[tuple] = None,
                          text_length: int = 1000,
                          query: str = "") -> List[Entity]:
    """
    Enhanced version that stores individual metric scores
    """

    def is_in_range(pos, ranges):
        """Check if position is within any of the given ranges"""
        if not ranges:
            return False
        return any(start <= pos <= end for start, end in ranges)

    def calculate_bm25_score(entity_text, query):
        """Improved BM25-like scoring with better matching"""
        if not query:
            return 0.0
        query_terms = query.lower().split()
        entity_text_lower = entity_text.lower()

        score = 0.0
        for term in query_terms:
            # Exact substring match
            if term in entity_text_lower:
                score += 1.0
            # Partial word match (for compound terms)
            elif any(term in word for word in entity_text_lower.split()):
                score += 0.5
            # Fuzzy match for similar terms
            elif any(abs(len(term) - len(word)) <= 2 and 
                    set(term) & set(word) for word in entity_text_lower.split()):
                score += 0.3

        return score / len(query_terms) if query_terms else 0.0

    # Calculate scores for all entities
    scores = []
    updated_entities = []

    for entity in entities:
        # Check contexts
        in_code = is_in_range(entity.start, code_positions)
        in_error = is_in_range(entity.start, error_positions)

        # Calculate BM25 score
        bm25_score = calculate_bm25_score(entity.text, query)

        # Calculate position score
        position_in_text = entity.start / text_length if text_length > 0 else 0

        # Calculate weighted score
        score = (
            weights.get('type', {}).get(entity.label, 0.0) +
            weights.get('code_context', 0.0) * (1 if in_code else 0) +
            weights.get('error_proximity', 0.0) * (1 if in_error else 0) +
            weights.get('bm25', 0.0) * bm25_score +
            weights.get('position', 0.0) * (1 - position_in_text)
        )

        scores.append(score)

    # Normalize scores with softmax
    scores = np.array(scores)
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    sum_exp_scores = np.sum(exp_scores)

    # Create updated entities with all metrics stored
    for i, entity in enumerate(entities):
        confidence = exp_scores[i] / sum_exp_scores
        
        # Calculate individual metrics
        in_code = is_in_range(entity.start, code_positions)
        in_error = is_in_range(entity.start, error_positions)
        bm25_score = calculate_bm25_score(entity.text, query)
        
        updated_entity = Entity(
            text=entity.text,
            label=entity.label,
            start=entity.start,
            end=entity.end,
            source=entity.source,
            confidence=confidence,
            bm25_score=bm25_score,  # Now this field exists in the dataclass
            in_code=in_code,
            near_error=in_error
        )
        updated_entities.append(updated_entity)

    return updated_entities

def _process_entities(updated_entities, problem_stmt, weights):
    """Process entities - now much simpler since metrics are pre-calculated"""
    processed = []
    
    for i, entity in enumerate(updated_entities):
        entity_dict = {
            'rank': i + 1,
            'text': entity.text,
            'label': entity.label,
            'start': entity.start,
            'end': entity.end,
            'source': entity.source,
            'confidence': entity.confidence,
            'length': len(entity.text),
            'position_normalized': entity.start / len(problem_stmt) if problem_stmt else 0,
            'in_code': entity.in_code,  # Use pre-calculated value
            'near_error': entity.near_error,  # Use pre-calculated value
            'bm25_score': entity.bm25_score,  # Use pre-calculated value
            'type_weight': weights.get('type', {}).get(entity.label.lower(), 0.0) if isinstance(weights, dict) else 0.0
        }
        processed.append(entity_dict)
        
    return processed

def initialize_data():
    """Initialize all the data needed for visualization"""
    weights = {
        'type': {
            'module': 0.05,
            'function': 0.07,
            'class': 0.05,
            'variable': 0.07,
            'path': 0.05,
            'example': 0.04,
            'constraint': 0.02
        },
        'code_context': 0.25,
        'error_proximity': 0.35,
        'bm25': 0.3,
        'position': 0.1
    }

    # Get entities from extraction module
    entities, extractor = extract_main()

    # Extract code and error positions
    code_positions, error_positions = extract_code_and_error_positions(problem_stmt)

    # Generate query automatically from problem statement
    auto_query = generate_query_from_problem(problem_stmt)

    # Calculate corrected scores
    updated_entities = calculate_entity_scores(
        entities=entities,
        weights=weights,
        code_positions=code_positions,
        error_positions=error_positions,
        text_length=len(problem_stmt),
        query=auto_query
    )
    updated_entities = sorted(updated_entities, key=attrgetter('confidence'), reverse=True)

    return entities, updated_entities, weights, code_positions, error_positions, auto_query

# Initialize data at module level so it can be imported
entities, updated_entities, weights, code_positions, error_positions, auto_query = initialize_data()

# Main execution
if __name__ == "__main__":
    # Print results
    print(f"Auto-generated query: '{auto_query}'")
    print("Updated entities with corrected scores:")
    for entity in updated_entities:
        print(f"Entity(text='{entity.text}', label='{entity.label}', "
              f"confidence={entity.confidence:.6f})")

    # Verify that confidences sum to 1
    total_confidence = sum(entity.confidence for entity in updated_entities)
    print(f"\nTotal confidence sum: {total_confidence:.6f}")