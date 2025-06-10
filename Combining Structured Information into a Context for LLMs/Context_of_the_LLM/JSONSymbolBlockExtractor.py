import json
import re
from typing import List, Dict, Union
from collections import defaultdict

# Import the ranking system and entities from the first file
from ranking_entities import (
    Entity as RankedEntity,  # Use the dataclass Entity from ranking.py
    updated_entities,        # Pre-calculated ranked entities
    weights,                # Scoring weights
    _process_entities,      # Entity processing function
    initialize_data         # Data initialization function
)


class EnhancedJSONSymbolBlockExtractor:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.data = self._load_json()
        self.matched_blocks = defaultdict(list)
        
        # Initialize ranking data
        self.entities, self.updated_entities, self.weights, _, _, _ = initialize_data()

    def _load_json(self):
        """Load JSON data from file."""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_symbol_parts(self, symbol: str) -> Dict[str, str]:
        """Extract different parts of a SCIP symbol for matching."""
        parts = {
            'full_symbol': symbol,
            'module_path': '',
            'class_name': '',
            'function_name': '',
            'function_with_params': '',
            'variable_name': '',
            'end_part': ''
        }

        # Extract the end part after the last `/` or backtick
        if '`/' in symbol:
            end_part = symbol.split('`/')[-1].rstrip('.')
            # Remove trailing # and everything after it for class names
            if '#' in end_part:
                end_part = end_part.split('#')[0]
            parts['end_part'] = end_part
        elif '/' in symbol:
            end_part = symbol.split('/')[-1].rstrip('.')
            # Remove trailing # and everything after it for class names
            if '#' in end_part:
                end_part = end_part.split('#')[0]
            parts['end_part'] = end_part

        # Extract module path (everything between backticks)
        if '`' in symbol:
            backtick_content = symbol.split('`')[1] if len(symbol.split('`')) > 1 else ''
            if '`/' in symbol:
                parts['module_path'] = backtick_content.split('/')[0]
            else:
                parts['module_path'] = backtick_content

        # Extract function name and parameters
        if '()' in symbol or '(' in symbol:
            func_part = symbol.split('/')[-1]
            if '(' in func_part:
                # Extract function name with parameters
                parts['function_with_params'] = func_part
                # Extract just the function name (before opening parenthesis)
                parts['function_name'] = func_part.split('(')[0].split('.')[-1]
            else:
                parts['function_name'] = func_part.replace('()', '').split('.')[-1]

        # Extract class name - look for patterns that indicate class definitions
        # Check for class name in the end part (after removing #)
        if parts['end_part'] and not '(' in parts['end_part']:
            # If it looks like a class name (starts with uppercase)
            if parts['end_part'][0].isupper():
                parts['class_name'] = parts['end_part']

        return parts

    def _parse_function_entity(self, entity_text: str) -> Dict[str, str]:
        """Parse function entity to extract name and parameters."""
        result = {
            'name': entity_text,
            'has_params': False,
            'params': '',
            'full_signature': entity_text
        }

        # Check if entity contains parentheses
        if '(' in entity_text and ')' in entity_text:
            result['has_params'] = True
            match = re.match(r'([^(]+)\(([^)]*)\)', entity_text)
            if match:
                result['name'] = match.group(1).strip()
                result['params'] = match.group(2).strip()
                result['full_signature'] = entity_text

        return result

    def _match_function_entity(self, entity_text: str, symbol: str) -> bool:
        """Enhanced function matching with parameter consideration."""
        func_info = self._parse_function_entity(entity_text)
        symbol_parts = self._extract_symbol_parts(symbol)

        # If entity has parameters, look for exact match
        if func_info['has_params']:
            return (
                func_info['full_signature'] in symbol or
                symbol.endswith(f"/{func_info['full_signature']}") or
                symbol.endswith(f".{func_info['full_signature']}") or
                # Match with function name and parameters
                (func_info['name'] == symbol_parts['function_name'] and
                 func_info['params'] in symbol_parts['function_with_params'])
            )
        else:
            # If no parameters, look for function with empty parentheses or just name
            return (
                func_info['name'] == symbol_parts['function_name'] or
                symbol.endswith(f"/{func_info['name']}()") or
                symbol.endswith(f".{func_info['name']}()") or
                symbol.endswith(f"/{func_info['name']}") or
                symbol.endswith(f".{func_info['name']}")
            )

    def _parse_example_entity(self, entity_text: str) -> Dict[str, str]:
        """Parse example entity to extract function calls and their parameters."""
        result = {
            'original': entity_text,
            'function_name': '',
            'has_complex_params': False,
            'params': '',
            'should_match_exact': False
        }

        # Match function call pattern: function_name(parameters)
        func_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)', entity_text)
        if func_match:
            result['function_name'] = func_match.group(1)
            result['params'] = func_match.group(2).strip()

            # Check if parameters are complex (contain function calls, operators, etc.)
            if any(char in result['params'] for char in ['&', '|', '.', '(', ')', 'Sky', 'TAN', 'Linear']):
                result['has_complex_params'] = True
                result['should_match_exact'] = True
            elif result['params']:  # Simple parameters
                result['should_match_exact'] = True
        else:
            # Just a function name without parentheses
            result['function_name'] = entity_text

        return result

    def _match_example_entity(self, entity_text: str, symbol: str) -> bool:
        """Enhanced example matching that looks for function calls with specific contexts."""
        example_info = self._parse_example_entity(entity_text)
        symbol_parts = self._extract_symbol_parts(symbol)

        # For examples with complex parameters, we need to be more restrictive
        if example_info['has_complex_params']:
            # Only match if it's the basic function definition, not a specific parameterized call
            # Since SCIP symbols represent code structure, not runtime calls
            return (
                example_info['function_name'] == symbol_parts['function_name'] and
                # Make sure it's the main function definition, not a method call
                symbol.endswith(f"/{example_info['function_name']}()") and
                # Exclude overly specific method calls
                not any(specific in symbol.lower() for specific in ['method', 'call', 'invoke'])
            )
        else:
            # For simple examples, use standard function matching
            return self._match_function_entity(entity_text, symbol)

    def _is_relevant_path_match(self, entity_text: str, symbol: str) -> bool:
        """Check if a path match is actually relevant and not just noise."""
        # For very broad path searches like 'astropy.modeling', be more restrictive
        if '.' in entity_text:
            parts = entity_text.split('.')
            # Must have the exact module path structure
            return (
                # Exact module path match
                entity_text in symbol and
                # Must be in the module definition part (backticks)
                '`' + entity_text in symbol
            ) or (
                # Or exact match in the symbol
                symbol.endswith(entity_text) or
                f'`{entity_text}`' in symbol
            )
        else:
            # Single word path - more lenient
            return entity_text in symbol

    def _match_entity_to_symbol(self, entity: Union[RankedEntity, Dict, str], symbol: str) -> bool:
        """Determine if an entity matches a symbol based on entity label and text."""
        # Handle RankedEntity (dataclass from ranking.py)
        if isinstance(entity, RankedEntity):
            entity_text = entity.text.strip()
            entity_label = entity.label.lower()
        # Handle dict format
        elif isinstance(entity, dict):
            entity_text = entity.get('text', '').strip()
            entity_label = entity.get('label', '').lower()
        # Handle string format
        else:
            entity_text = str(entity).strip()
            entity_label = 'unknown'

        symbol_parts = self._extract_symbol_parts(symbol)

        # Exact matching strategies based on entity label
        if entity_label == 'function':
            return self._match_function_entity(entity_text, symbol)

        elif entity_label == 'class':
            # For classes, match against class name patterns - be more precise
            return (
                entity_text == symbol_parts['class_name'] or
                entity_text == symbol_parts['end_part'] or
                # Check if it's a direct class reference
                (symbol.endswith(f'/{entity_text}#') or
                 symbol.endswith(f'/{entity_text}') or
                 f'/{entity_text}#' in symbol) and
                # Ensure it's actually a class definition, not just a substring
                not any(noise in symbol.lower() for noise in ['__init__', 'version', 'display'])
            )

        elif entity_label == 'variable':
            # For variables, match against end part or variable patterns
            return (
                entity_text == symbol_parts['end_part'] or
                entity_text == symbol_parts['variable_name'] or
                symbol.endswith(f"/{entity_text}") or
                symbol.endswith(f".{entity_text}") or
                # Handle variable assignments
                f"{entity_text}:" in symbol
            )

        elif entity_label in ['path', 'module']:
            # For paths/modules, be much more restrictive to avoid noise
            return self._is_relevant_path_match(entity_text, symbol)

        elif entity_label == 'example':
            # Use specialized example matching
            return self._match_example_entity(entity_text, symbol)

        # Fallback: enhanced partial match
        return (
            entity_text.lower() in symbol.lower() or
            entity_text in symbol_parts['end_part'] or
            any(word in symbol.lower() for word in entity_text.lower().split('_'))
        )

    def _recurse_blocks(self, obj, entity):
        """Recursively find blocks that match the entity."""
        if isinstance(obj, dict):
            if "symbol" in obj and isinstance(obj["symbol"], str):
                if self._match_entity_to_symbol(entity, obj["symbol"]):
                    yield obj
            for v in obj.values():
                yield from self._recurse_blocks(v, entity)
        elif isinstance(obj, list):
            for item in obj:
                yield from self._recurse_blocks(item, entity)

    def extract_blocks_from_ranked_entities(self, top_n: int = 10):
        """Extract blocks using the top N ranked entities from ranking.py."""
        self.matched_blocks.clear()
        
        # Use the top N entities based on confidence scores
        top_entities = self.updated_entities[:top_n]
        
        print(f"Using top {len(top_entities)} ranked entities for block extraction:")
        for i, entity in enumerate(top_entities):
            print(f"  {i+1}. {entity.text} ({entity.label}) - confidence: {entity.confidence:.4f}")
        
        # Extract blocks for each entity
        for entity in top_entities:
            entity_key = f"{entity.text} ({entity.label})"
            
            # Collect all matches
            all_matches = list(self._recurse_blocks(self.data, entity))
            
            # For path/module entities, limit results to avoid overwhelming output
            if entity.label.lower() in ['path', 'module']:
                all_matches = all_matches[:20]  # Limit to 20 most relevant
            
            self.matched_blocks[entity_key].extend(all_matches)
        
        return self.matched_blocks

    def extract_blocks(self, entities: List[Union[RankedEntity, Dict, str]]):
        """Extract blocks for a list of entities (RankedEntity, dict, or strings)."""
        self.matched_blocks.clear()
        for entity in entities:
            # Create a key for storing results
            if isinstance(entity, RankedEntity):
                entity_key = f"{entity.text} ({entity.label})"
            elif isinstance(entity, dict):
                entity_key = f"{entity.get('text', entity)} ({entity.get('label', 'unknown')})"
            else:
                entity_key = str(entity)

            # Collect all matches first
            all_matches = list(self._recurse_blocks(self.data, entity))

            # For path/module entities, limit results to avoid overwhelming output
            if ((isinstance(entity, RankedEntity) and entity.label.lower() in ['path', 'module']) or
                (isinstance(entity, dict) and entity.get('label', '').lower() in ['path', 'module'])):
                all_matches = all_matches[:20]  # Limit to 20 most relevant

            self.matched_blocks[entity_key].extend(all_matches)

    def get_blocks(self, entity_key: str) -> List[Dict]:
        """Get blocks for a specific entity key."""
        return self.matched_blocks.get(entity_key, [])

    def save_to_file(self, output_file: str):
        """Save matched blocks to a JSON file."""
        output = {
            entity: blocks
            for entity, blocks in self.matched_blocks.items()
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4)
        print(f"\nSaved matched blocks to '{output_file}'")

    def print_detailed_matches(self, entities: List[Union[RankedEntity, Dict, str]] = None, 
                             show_details: bool = False):
        """Print detailed matching results with explanations."""
        if entities is None:
            # Use the matched blocks already computed
            entities_to_show = []
            for entity_key in self.matched_blocks.keys():
                # Parse the key to recreate entity info
                if ' (' in entity_key and entity_key.endswith(')'):
                    text = entity_key.split(' (')[0]
                    label = entity_key.split(' (')[1].rstrip(')')
                    entities_to_show.append({'text': text, 'label': label})
                else:
                    entities_to_show.append({'text': entity_key, 'label': 'unknown'})
            entities = entities_to_show

        for entity in entities:
            # Create entity key and extract info
            if isinstance(entity, RankedEntity):
                entity_text = entity.text
                entity_label = entity.label
                entity_key = f"{entity_text} ({entity_label})"
                confidence = entity.confidence
            elif isinstance(entity, dict):
                entity_text = entity.get('text', str(entity))
                entity_label = entity.get('label', 'unknown')
                entity_key = f"{entity_text} ({entity_label})"
                confidence = entity.get('confidence', 0.0)
            else:
                entity_text = str(entity)
                entity_label = 'string'
                entity_key = entity_text
                confidence = 0.0

            blocks = self.get_blocks(entity_key)
            
            # Show confidence score if available
            conf_str = f" - Confidence: {confidence:.4f}" if confidence > 0 else ""
            print(f"\n=== Matches for '{entity_text}' [Label: {entity_label}]{conf_str} ({len(blocks)} found) ===")

            if not blocks:
                print("No matches found.")
                continue

            # Show more matches for specific types, fewer for broad searches
            max_display = 3 if entity_label in ['path', 'module'] else 5

            for i, block in enumerate(blocks[:max_display]):
                print(f"\nMatch {i+1}:")
                print(f"Symbol: {block['symbol']}")
                if show_details:
                    print(f"Range: {block.get('range', 'N/A')}")
                    print(f"Symbol Roles: {block.get('symbol_roles', 'N/A')}")

                    # Show why it matched
                    symbol_parts = self._extract_symbol_parts(block['symbol'])
                    print(f"Parsed symbol parts: {symbol_parts}")

                    if entity_label == 'function':
                        func_info = self._parse_function_entity(entity_text)
                        print(f"Function info: {func_info}")

            if len(blocks) > max_display:
                print(f"\n...and {len(blocks)-max_display} more matches")

    def get_ranked_entity_summary(self):
        """Get summary of ranked entities and their scores."""
        print("\n" + "="*60)
        print("RANKED ENTITY SUMMARY")
        print("="*60)
        
        processed_entities = _process_entities(self.updated_entities, "", self.weights)
        
        for i, entity_dict in enumerate(processed_entities[:15]):  # Show top 15
            print(f"\nRank {entity_dict['rank']}: {entity_dict['text']} ({entity_dict['label']})")
            print(f"  Confidence: {entity_dict['confidence']:.6f}")
            print(f"  BM25 Score: {entity_dict['bm25_score']:.3f}")
            print(f"  In Code: {entity_dict['in_code']}")
            print(f"  Near Error: {entity_dict['near_error']}")
            print(f"  Type Weight: {entity_dict['type_weight']:.3f}")


# Example usage and main execution
if __name__ == "__main__":
    try:
        # Initialize extractor with JSON file
        extractor = EnhancedJSONSymbolBlockExtractor("Context_of_the_LLM/formatted_output.json")
        
        # Show summary of ranked entities
        extractor.get_ranked_entity_summary()
        
        # Method 1: Use top ranked entities automatically
        print("\n" + "="*60)
        print("EXTRACTING BLOCKS FROM TOP RANKED ENTITIES")
        print("="*60)
        
        extractor.extract_blocks_from_ranked_entities(top_n=8)
        extractor.print_detailed_matches(show_details=True)
        
        # Method 2: Use specific entities if you want custom selection
        # custom_entities = extractor.updated_entities[:5]  # Top 5 entities
        # extractor.extract_blocks(custom_entities)
        # extractor.print_detailed_matches(custom_entities, show_details=True)
        
        # Save results to file
        extractor.save_to_file("matched_blocks_ranked.json")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure 'formatted_output.json' exists in the current directory.")
        print("Also ensure that ranking.py and its dependencies are available.")
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Make sure all required modules are properly imported and configured.")