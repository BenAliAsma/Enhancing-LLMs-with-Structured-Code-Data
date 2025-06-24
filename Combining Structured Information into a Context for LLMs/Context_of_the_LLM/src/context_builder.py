import json
import os
from collections import defaultdict
from typing import Dict, Any, List, Union
from src.config import STRATEGIES


class ContextBuilder:
    """Builds different types of contexts for LLM patch generation"""
    
    def __init__(self):
        self.strategies = {
            'minimal': self._build_minimal_context,
            'balanced': self._build_balanced_context,
            'comprehensive': self._build_comprehensive_context,
            'rag_style': self._build_rag_context,
            'hierarchical': self._build_hierarchical_context
        }
        
        # Path to the matched blocks file
        self.matched_blocks_path = "/home/abenali/Enhancing-LLMs-with-Structured-Code-Data/Combining Structured Information into a Context for LLMs/matched_blocks_ranked.json"
        self.matched_blocks_data = self._load_matched_blocks()
        self.code_snippets = self._load_code_snippets()
    
    def _get_entity_info(self, entity) -> Dict[str, Any]:
        """
        Extract standardized information from an entity object.
        Handles different entity formats and provides consistent output.
        """
        try:
            # Handle different entity formats
            if hasattr(entity, '__dict__'):
                # Entity object with attributes
                info = {
                    'name': getattr(entity, 'text', str(entity)),
                    'type': getattr(entity, 'label', 'unknown'),
                    'start': getattr(entity, 'start', 0),
                    'end': getattr(entity, 'end', 0),
                    'confidence': float(getattr(entity, 'confidence', 0.0)),
                    'bm25_score': float(getattr(entity, 'bm25_score', 0.0)),
                    'in_code': getattr(entity, 'in_code', False),
                    'near_error': getattr(entity, 'near_error', False),
                    'source': getattr(entity, 'source', 'unknown')
                }
            elif isinstance(entity, dict):
                # Dictionary format
                info = {
                    'name': entity.get('text', entity.get('name', str(entity))),
                    'type': entity.get('label', entity.get('type', 'unknown')),
                    'start': entity.get('start', 0),
                    'end': entity.get('end', 0),
                    'confidence': float(entity.get('confidence', 0.0)),
                    'bm25_score': float(entity.get('bm25_score', 0.0)),
                    'in_code': entity.get('in_code', False),
                    'near_error': entity.get('near_error', False),
                    'source': entity.get('source', 'unknown')
                }
            else:
                # Fallback for unknown formats
                info = {
                    'name': str(entity),
                    'type': 'unknown',
                    'start': 0,
                    'end': 0,
                    'confidence': 0.0,
                    'bm25_score': 0.0,
                    'in_code': False,
                    'near_error': False,
                    'source': 'unknown'
                }
            
            # Add derived fields
            info['length'] = info['end'] - info['start']
            info['rank'] = getattr(entity, 'rank', 0)
            
            return info
            
        except Exception as e:
            # Robust fallback
            return {
                'name': str(entity)[:50],  # Truncate long names
                'type': 'error',
                'start': 0,
                'end': 0,
                'length': 0,
                'confidence': 0.0,
                'bm25_score': 0.0,
                'in_code': False,
                'near_error': False,
                'source': 'error',
                'rank': 0,
                'error': str(e)
            }
    
    def _format_matched_block_info(self, block) -> Dict[str, str]:
        """
        Format matched block information for display in contexts.
        Extracts key information from matched block data structure.
        """
        try:
            if isinstance(block, dict):
                # Extract information from block dictionary
                location = block.get('location', 'unknown')
                symbol = block.get('symbol', block.get('name', 'unknown'))
                full_symbol = block.get('full_symbol', block.get('full_name', symbol))
                roles = block.get('roles', [])
                enclosing_range = block.get('enclosing_range', '')
                
                # Handle different role formats
                if isinstance(roles, list):
                    roles_str = ', '.join(str(role) for role in roles)
                else:
                    roles_str = str(roles)
                
                return {
                    'location': str(location),
                    'symbol': str(symbol),
                    'full_symbol': str(full_symbol),
                    'roles': roles_str,
                    'enclosing_range': str(enclosing_range)
                }
                
            elif hasattr(block, '__dict__'):
                # Handle object with attributes
                location = getattr(block, 'location', getattr(block, 'file', 'unknown'))
                symbol = getattr(block, 'symbol', getattr(block, 'name', 'unknown'))
                full_symbol = getattr(block, 'full_symbol', getattr(block, 'full_name', symbol))
                roles = getattr(block, 'roles', [])
                enclosing_range = getattr(block, 'enclosing_range', '')
                
                if isinstance(roles, list):
                    roles_str = ', '.join(str(role) for role in roles)
                else:
                    roles_str = str(roles)
                    
                return {
                    'location': str(location),
                    'symbol': str(symbol),
                    'full_symbol': str(full_symbol),
                    'roles': roles_str,
                    'enclosing_range': str(enclosing_range)
                }
            else:
                # Fallback for unknown formats
                return {
                    'location': 'unknown',
                    'symbol': str(block)[:50],
                    'full_symbol': str(block)[:100],
                    'roles': 'unknown',
                    'enclosing_range': ''
                }
                
        except Exception as e:
            # Error handling fallback
            return {
                'location': 'error',
                'symbol': 'error',
                'full_symbol': 'error',
                'roles': f'error: {str(e)}',
                'enclosing_range': ''
            }
    
    def _load_matched_blocks(self) -> Dict[str, Any]:
        """Load matched blocks data from the JSON file"""
        try:
            if os.path.exists(self.matched_blocks_path):
                with open(self.matched_blocks_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✓ Loaded matched blocks data with {len(data)} entries")
                return data
            else:
                print(f"⚠ Matched blocks file not found: {self.matched_blocks_path}")
                return {}
        except Exception as e:
            print(f"⚠ Error loading matched blocks: {str(e)}")
            return {}

    def _load_code_snippets(self) -> str:
        """Load code snippets from the consolidated markdown file"""
        snippets_path = "/home/abenali/Enhancing-LLMs-with-Structured-Code-Data/Combining Structured Information into a Context for LLMs/outputs/all_snippets_consolidated.md"
        try:
            if os.path.exists(snippets_path):
                with open(snippets_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"✓ Loaded code snippets from {snippets_path}")
                return content
            else:
                print(f"⚠ Snippets file not found: {snippets_path}")
                return ""
        except Exception as e:
            print(f"⚠ Error loading code snippets: {str(e)}")
            return ""
    
    def build_all_contexts(self, problem_statement: str, entities: List[Dict], 
                          call_hierarchy: List, structured_data: Dict) -> Dict[str, str]:
        """Build contexts using all strategies"""
        contexts = {}
        
        for strategy in STRATEGIES:
            try:
                if strategy in self.strategies:
                    contexts[strategy] = self.strategies[strategy](
                        problem_statement, entities, call_hierarchy, structured_data
                    )
                else:
                    # Fallback to balanced context for unknown strategies
                    print(f"⚠ Unknown strategy '{strategy}', using balanced context")
                    contexts[strategy] = self._build_balanced_context(
                        problem_statement, entities, call_hierarchy, structured_data
                    )
            except Exception as e:
                print(f"⚠ Error building context for strategy '{strategy}': {str(e)}")
                # Provide a basic fallback context
                contexts[strategy] = self._build_fallback_context(problem_statement, entities)
        
        return contexts
    
    def _build_fallback_context(self, problem_statement: str, entities: List[Dict]) -> str:
        """Build a basic fallback context when other strategies fail"""
        context = f"""# SWE Benchmark Issue (Fallback Context)

## Problem
{problem_statement}

## Available Entities
"""
        try:
            for i, entity in enumerate(entities[:5]):
                entity_info = self._get_entity_info(entity)
                context += f"- {entity_info['name']} ({entity_info['type']})\n"
        except Exception as e:
            context += f"Error processing entities: {str(e)}\n"
            context += f"Raw entities: {str(entities[:3])}\n"
        
        context += "\n## Task\nGenerate a git diff patch to fix this issue.\n"
        return context
    
    def _build_balanced_context(self, problem_statement: str, entities: List[Dict], 
                           call_hierarchy: List, structured_data: Dict) -> str:
        """
        Balanced context optimized for structured code data integration.
        Focuses on key entities, relationships, and code structure without overwhelming the LLM.
        """
        context = f"""# Code Issue Analysis - Balanced Context

## Problem Statement
{problem_statement}

## Key Code Entities (Prioritized)
"""
        
        try:
            # Group entities by relevance and type for better organization
            high_priority_entities = []
            code_entities = []
            
            for entity in entities[:15]:  # Focus on top 15 most relevant
                info = self._get_entity_info(entity)
                if info['confidence'] > 0.2 or info['in_code'] or info['near_error']:
                    high_priority_entities.append(info)
                elif info['type'] in ['function', 'class', 'method', 'variable']:
                    code_entities.append(info)
            
            # Display high-priority entities first
            if high_priority_entities:
                context += "### High Priority Entities\n"
                for info in high_priority_entities[:8]:
                    context += f"• **{info['name']}** ({info['type']})\n"
                    context += f"  Confidence: {info['confidence']:.3f}"
                    if info['in_code']:
                        context += " | In codebase"
                    if info['near_error']:
                        context += " | Near error location"
                    context += f"\n  BM25: {info['bm25_score']:.3f} \n\n"
            
            # Display relevant code entities
            if code_entities:
                context += "### Code Structure Elements\n"
                for info in code_entities[:5]:
                    context += f"• {info['name']} ({info['type']}) - Confidence: {info['confidence']:.3f}\n"
                context += "\n"
                
        except Exception as e:
            context += f"Error processing entities: {str(e)}\n\n"
        
        # Add structured call relationships
        try:
            if call_hierarchy:
                context += "## Code Dependencies & Call Graph\n"
                # Process and deduplicate call relationships
                unique_calls = set()
                for item in call_hierarchy[:12]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        call_pair = (str(item[0]), str(item[1]))
                        unique_calls.add(call_pair)
                
                if unique_calls:
                    for caller, callee in sorted(unique_calls):
                        context += f"• {caller} → {callee}\n"
                    context += "\n"
        except Exception as e:
            context += f"Error processing call hierarchy: {str(e)}\n\n"
        
        # Add matched code blocks with structure
        try:
            matched_blocks = self._get_matched_blocks_for_entities(entities, max_blocks=3)
            if matched_blocks:
                context += "## Relevant Code Locations\n"
                for entity_key, blocks in list(matched_blocks.items())[:4]:
                    context += f"### {entity_key}\n"
                    for i, block in enumerate(blocks):
                        block_info = self._format_matched_block_info(block)
                        context += f"• Symbol: {block_info['symbol']}\n"
                    context += "\n"
        except Exception as e:
            context += f"Error processing matched blocks: {str(e)}\n\n"
        
        # Add concise code context
        try:
            if self.code_snippets:
                context += "## Code Context Sample\n```python\n"
                # Extract first 800 characters for balanced context
                snippet_preview = self.code_snippets[:800]
                if len(self.code_snippets) > 800:
                    snippet_preview += "\n# ... (truncated for brevity)"
                context += snippet_preview + "\n```\n\n"
        except Exception as e:
            context += f"Error adding code context: {str(e)}\n\n"
        
        context += "## Task\nAnalyze the structured information above and generate a precise patch to resolve the issue.\n"
        return context

    def _build_comprehensive_context(self, problem_statement: str, entities: List[Dict], 
                            call_hierarchy: List, structured_data: Dict) -> str:
        """
        Comprehensive context with full structured data integration.
        Maximizes information density while maintaining logical organization.
        """
        context = f"""# Comprehensive Code Analysis - Full Context

## Problem Statement
{problem_statement}

## Complete Entity Analysis with Structural Information
"""
        
        try:
            # Organize entities by type and confidence for comprehensive view
            entity_groups = defaultdict(list)
            for i, entity in enumerate(entities):
                info = self._get_entity_info(entity)
                info['index'] = i + 1
                entity_groups[info['type']].append(info)
            
            # Display entities grouped by type
            for entity_type, entities_list in entity_groups.items():
                if entities_list:
                    context += f"\n### {entity_type.upper()} Entities\n"
                    for info in sorted(entities_list, key=lambda x: x['confidence'], reverse=True):
                        context += f"{info['index']}. **{info['name']}**\n"
                        context += f"   • Confidence: {info['confidence']:.4f} | Rank: {info['rank']}\n"
                        context += f"   • Position: {info['start']}-{info['end']} | Length: {info['length']}\n"
                        context += f"   • BM25: {info['bm25_score']:.4f} \n"
                        
                        status_flags = []
                        if info['in_code']:
                            status_flags.append("In Codebase")
                        if info['near_error']:
                            status_flags.append("Near Error")
                        if status_flags:
                            context += f"   • Status: {' | '.join(status_flags)}\n"
                        context += "\n"
        except Exception as e:
            context += f"Error in entity analysis: {str(e)}\n\n"
        
        # Complete call hierarchy with structure analysis
        try:
            if call_hierarchy:
                context += "## Complete Call Graph & Dependencies\n"
                
                # Build call graph structure
                call_graph = defaultdict(set)
                reverse_graph = defaultdict(set)
                
                for item in call_hierarchy:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        caller, callee = str(item[0]), str(item[1])
                        call_graph[caller].add(callee)
                        reverse_graph[callee].add(caller)
                
                # Display call relationships
                context += "### Function Call Relationships\n"
                for caller, callees in sorted(call_graph.items()):
                    context += f"**{caller}** calls:\n"
                    for callee in sorted(callees):
                        context += f"  → {callee}\n"
                    context += "\n"
                
                # Display dependency analysis
                context += "### Dependency Analysis\n"
                root_functions = set(call_graph.keys()) - set(reverse_graph.keys())
                leaf_functions = set(reverse_graph.keys()) - set(call_graph.keys())
                
                if root_functions:
                    context += f"Root functions (no callers): {', '.join(sorted(root_functions))}\n"
                if leaf_functions:
                    context += f"Leaf functions (no callees): {', '.join(sorted(leaf_functions))}\n"
                context += "\n"
        except Exception as e:
            context += f"Error in call hierarchy analysis: {str(e)}\n\n"
        
        # Comprehensive matched blocks analysis
        try:
            matched_blocks = self._get_matched_blocks_for_entities(entities, max_blocks=8)
            if matched_blocks:
                context += "## Detailed Code Block Analysis\n"
                for entity_key, blocks in matched_blocks.items():
                    context += f"### {entity_key}\n"
                    for i, block in enumerate(blocks):
                        block_info = self._format_matched_block_info(block)
                        context += f"**Block {i+1}**:\n"
                        context += f"  • Symbol: {block_info['symbol']}\n"
                        context += f"  • Full Path: {block_info['full_symbol']}\n"
                        if block_info['enclosing_range']:
                            context += f"  • Enclosing: {block_info['enclosing_range']}\n"
                        context += "\n"
            else:
                context += "## Code Block Analysis\nNo matched blocks found for current entities.\n\n"
        except Exception as e:
            context += f"Error in matched blocks analysis: {str(e)}\n\n"
        
        # Full code context
        try:
            if self.code_snippets:
                context += "## Complete Code Context\n"
                context += "```python\n"
                context += self.code_snippets
                context += "\n```\n\n"
        except Exception as e:
            context += f"Error adding complete code context: {str(e)}\n\n"
        
        # Additional structured data
        try:
            if structured_data:
                context += "## Additional Structured Information\n"
                for key, data in structured_data.items():
                    if key not in ['metadata', 'all_snippets_combined'] and data:
                        context += f"### {key.replace('_', ' ').title()}\n"
                        if isinstance(data, dict):
                            for sub_key, sub_value in list(data.items())[:5]:
                                context += f"• {sub_key}: {str(sub_value)[:100]}\n"
                        elif isinstance(data, list):
                            for i, item in enumerate(data[:5]):
                                context += f"• Item {i+1}: {str(item)[:100]}\n"
                        else:
                            context += f"{str(data)[:200]}\n"
                        context += "\n"
        except Exception as e:
            context += f"Error processing additional data: {str(e)}\n\n"
        
        context += "## Task\nUsing all the comprehensive structural information above, provide detailed analysis and generate an optimal patch solution.\n"
        return context

    def _build_rag_context(self, problem_statement: str, entities: List[Dict], 
                        call_hierarchy: List, structured_data: Dict) -> str:
        """
        RAG-style context focusing on code retrieval and semantic similarity.
        Emphasizes code snippets and their semantic relationships to the problem.
        """
        context = f"""# Code Retrieval & Generation Context

## Issue Description
{problem_statement}

## Retrieved Code Snippets (Semantic Matching)
"""
        
        # Primary code context
        try:
            if self.code_snippets:
                context += "### Main Code Repository\n"
                context += "```python\n"
                context += self.code_snippets
                context += "\n```\n\n"
        except Exception as e:
            context += f"Error retrieving main code snippets: {str(e)}\n\n"
        
        # Semantically related code blocks
        try:
            matched_blocks = self._get_matched_blocks_for_entities(entities, max_blocks=6)
            if matched_blocks:
                context += "## Semantically Related Code Blocks\n"
                for entity_key, blocks in matched_blocks.items():
                    context += f"### Related to: {entity_key}\n"
                    for i, block in enumerate(blocks):
                        block_info = self._format_matched_block_info(block)
                        context += f"**Match {i+1}**: \n"
                        context += f"```python\n# {block_info['symbol']}\n```\n\n"
        except Exception as e:
            context += f"Error processing semantic matches: {str(e)}\n\n"
        
        # Entity-based code retrieval
        try:
            context += "## Key Symbols & Code Elements\n"
            
            # Group entities by confidence for retrieval ranking
            high_conf = [e for e in entities if self._get_entity_info(e)['confidence'] > 0.3]
            medium_conf = [e for e in entities if 0.1 < self._get_entity_info(e)['confidence'] <= 0.3]
            
            if high_conf:
                context += "### High Confidence Matches\n"
                for entity in high_conf[:8]:
                    info = self._get_entity_info(entity)
                    context += f"• **{info['name']}** ({info['type']}) - Score: {info['confidence']:.3f}\n"
                    if info['bm25_score'] > 0:
                        context += f"  BM25 Relevance: {info['bm25_score']:.3f}\n"
                context += "\n"
            
            if medium_conf:
                context += "### Medium Confidence Matches\n"
                for entity in medium_conf[:6]:
                    info = self._get_entity_info(entity)
                    context += f"• {info['name']} ({info['type']}) - Score: {info['confidence']:.3f}\n"
                context += "\n"
        except Exception as e:
            context += f"Error processing entity-based retrieval: {str(e)}\n\n"
        
        # Function call context for RAG
        try:
            if call_hierarchy:
                context += "## Function Call Context\n"
                context += "```\n# Call relationships (for context):\n"
                
                for item in call_hierarchy[:10]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        context += f"# {item[0]} → {item[1]}\n"
                context += "```\n\n"
        except Exception as e:
            context += f"Error adding call context: {str(e)}\n\n"
        
        # Additional retrieved context
        try:
            if structured_data.get('all_snippets_combined'):
                context += "## Additional Retrieved Context\n"
                additional_snippets = structured_data['all_snippets_combined']
                if isinstance(additional_snippets, str) and len(additional_snippets) > 0:
                    context += "```python\n"
                    context += additional_snippets[:1000]  # Limit for RAG context
                    if len(additional_snippets) > 1000:
                        context += "\n# ... (additional snippets available)"
                    context += "\n```\n\n"
        except Exception as e:
            context += f"Error adding additional context: {str(e)}\n\n"
        
        context += "## Generation Task\nBased on the retrieved code snippets and semantic matches above, generate a targeted patch that addresses the specific issue.\n"
        return context

    def _build_hierarchical_context(self, problem_statement: str, entities: List[Dict], 
                                call_hierarchy: List, structured_data: Dict) -> str:
        """
        Hierarchical context organized by code structure and architectural layers.
        Presents information in a top-down, structured manner following code organization principles.
        """
        context = f"""# Hierarchical Code Structure Analysis

## Problem Statement
{problem_statement}

## Code Architecture Overview
"""
        
        # Layer 1: High-level entity classification
        try:
            # Classify entities into architectural layers
            layers = {
                'interfaces': [],
                'classes': [],
                'functions': [],
                'variables': [],
                'modules': [],
                'other': []
            }
            
            for entity in entities:
                info = self._get_entity_info(entity)
                entity_type = info['type'].lower()
                
                if entity_type in ['interface', 'protocol']:
                    layers['interfaces'].append(info)
                elif entity_type in ['class', 'struct']:
                    layers['classes'].append(info)
                elif entity_type in ['function', 'method', 'procedure']:
                    layers['functions'].append(info)
                elif entity_type in ['variable', 'field', 'property']:
                    layers['variables'].append(info)
                elif entity_type in ['module', 'package', 'namespace']:
                    layers['modules'].append(info)
                else:
                    layers['other'].append(info)
            
            # Display architectural layers
            layer_names = {
                'modules': 'Module/Package Layer',
                'interfaces': 'Interface/Protocol Layer', 
                'classes': 'Class/Type Layer',
                'functions': 'Function/Method Layer',
                'variables': 'Variable/Field Layer',
                'other': 'Other Elements'
            }
            
            for layer_key, layer_name in layer_names.items():
                entities_in_layer = layers[layer_key]
                if entities_in_layer:
                    context += f"\n### {layer_name}\n"
                    # Sort by confidence within each layer
                    sorted_entities = sorted(entities_in_layer, key=lambda x: x['confidence'], reverse=True)
                    for info in sorted_entities[:8]:  # Limit per layer
                        context += f"• **{info['name']}**\n"
                        context += f"  Confidence: {info['confidence']:.3f} | Rank: {info['rank']}\n"
                        if info['in_code'] or info['near_error']:
                            status = []
                            if info['in_code']:
                                status.append("In Code")
                            if info['near_error']:
                                status.append("Near Error")
                            context += f"  Status: {' | '.join(status)}\n"
                        context += f"  BM25: {info['bm25_score']:.3f}\n\n"
                        
        except Exception as e:
            context += f"Error in architectural analysis: {str(e)}\n\n"
        
        # Layer 2: Call hierarchy organization
        try:
            if call_hierarchy:
                context += "## Function Call Hierarchy\n"
                
                # Build hierarchical call structure
                call_tree = defaultdict(list)
                all_callees = set()
                
                for item in call_hierarchy:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        caller, callee = str(item[0]), str(item[1])
                        call_tree[caller].append(callee)
                        all_callees.add(callee)
                
                # Find root functions (functions that call others but aren't called)
                root_functions = set(call_tree.keys()) - all_callees
                
                # Display hierarchical structure
                def display_call_tree(func, level=0, visited=None):
                    if visited is None:
                        visited = set()
                    if func in visited or level > 3:  # Prevent infinite recursion
                        return ""
                    
                    visited.add(func)
                    indent = "  " * level
                    result = f"{indent}• {func}\n"
                    
                    for callee in call_tree.get(func, []):
                        result += display_call_tree(callee, level + 1, visited.copy())
                    
                    return result
                
                if root_functions:
                    context += "### Call Trees (Top-Down)\n"
                    for root in sorted(root_functions):
                        context += display_call_tree(root)
                    context += "\n"
                else:
                    context += "### Function Relationships\n"
                    for caller, callees in sorted(call_tree.items()):
                        context += f"• **{caller}** calls:\n"
                        for callee in callees:
                            context += f"  → {callee}\n"
                    context += "\n"
                    
        except Exception as e:
            context += f"Error in call hierarchy analysis: {str(e)}\n\n"
        
        # Layer 3: Code structure with matched blocks
        try:
            matched_blocks = self._get_matched_blocks_for_entities(entities, max_blocks=5)
            if matched_blocks:
                context += "## Code Structure Mapping\n"
                
                # Organize blocks by location/file structure
                location_groups = defaultdict(list)
                for entity_key, blocks in matched_blocks.items():
                    for block in blocks:
                        block_info = self._format_matched_block_info(block)
                        location_groups[entity_key].append(block_info)
                
                for entity_key, block_infos in location_groups.items():
                    context += f"### {entity_key}\n"
                    for i, block_info in enumerate(block_infos):
                        context += f"  • Symbol: {block_info['symbol']}\n"
        except Exception as e:
            context += f"Error in code structure mapping: {str(e)}\n\n"
        
        # Layer 4: Implementation details
        try:
            if self.code_snippets:
                context += "## Implementation Layer\n"
                context += "### Core Code Implementation\n"
                context += "```python\n"
                # For hierarchical context, show code in structured way
                lines = self.code_snippets.split('\n')
                if len(lines) > 30:
                    # Show beginning and end for hierarchical overview
                    context += '\n'.join(lines[:15])
                    context += f"\n\n# ... ({len(lines) - 30} lines of implementation) ...\n\n"
                    context += '\n'.join(lines[-15:])
                else:
                    context += self.code_snippets
                context += "\n```\n\n"
        except Exception as e:
            context += f"Error in implementation layer: {str(e)}\n\n"
        
        # Layer 5: High-confidence elements summary
        try:
            high_confidence_entities = [
                e for e in entities 
                if self._get_entity_info(e)['confidence'] > 0.2
            ]
            
            if high_confidence_entities:
                context += "## High-Confidence Analysis Summary\n"
                context += "### Critical Elements (Confidence > 0.2)\n"
                for entity in high_confidence_entities[:8]:
                    info = self._get_entity_info(entity)
                    context += f"• **{info['name']}** ({info['type']}) - {info['confidence']:.3f}\n"
                    if info['near_error']:
                        context += f"  ⚠ Located near error context\n"
                context += "\n"
        except Exception as e:
            context += f"Error in confidence analysis: {str(e)}\n\n"
        
        context += "## Synthesis Task\nAnalyze the hierarchical code structure above and generate a well-structured patch that respects the architectural organization.\n"
        return context
    
    def _get_matched_blocks_for_entities(self, entities: List[Dict], max_blocks: int = 5) -> Dict[str, List]:
        """Get matched blocks for relevant entities"""
        matched_info = {}
        
        try:
            for entity in entities[:10]:  # Check top 10 entities
                entity_info = self._get_entity_info(entity)
                entity_name = entity_info['name']
                entity_type = entity_info['type']
                
                # Create key as it appears in matched_blocks_ranked.json
                key = f"{entity_name} ({entity_type})"
                
                if key in self.matched_blocks_data:
                    blocks = self.matched_blocks_data[key][:max_blocks]  # Limit blocks per entity
                    matched_info[key] = blocks
        except Exception as e:
            print(f"⚠ Error getting matched blocks: {str(e)}")
                
        return matched_info

        #the matched blocks already exists /home/abenali/Enhancing-LLMs-with-Structured-Code-Data/Combining Structured Information into a Context for LLMs/matched_blocks_ranked.json
    
    def _build_minimal_context(self, problem_statement: str, entities: List[Dict], 
                              call_hierarchy: List, structured_data: Dict) -> str:
        """Minimal context with only essential information"""
        context = f"""# SWE Benchmark Issue

## Problem
{problem_statement}
"""
        
        context += "\n## Task\nGenerate a git diff patch to fix this issue.\n"
        return context
    
    def _build_balanced_context(self, problem_statement: str, entities: List[Dict], 
                           call_hierarchy: List, structured_data: Dict) -> str:
        """
        Balanced context optimized for structured code data integration.
        Focuses on key entities, relationships, and code structure without overwhelming the LLM.
        """
        context = f"""# Code Issue Analysis - Balanced Context

    ## Problem Statement
    {problem_statement}

    ## Key Code Entities (Prioritized)
    """
    #print(updated_entities) [Entity(text='separability_matrix', label='function', start=12, end=31, source='gliner', confidence=np.float64(0.12985493596005734), bm25_score=0.3333333333333333, in_code=False, near_error=True), Entity(text='astropy.modeling', label='path', start=143, end=183, source='regex', confidence=np.float64(0.11396647275640677), bm25_score=0.3333333333333333, in_code=True, near_error=False), Entity(text='astropy.modeling.separable', label='path', start=185, end=243, source='regex', confidence=np.float64(0.11358296283482879), bm25_score=0.3333333333333333, in_code=True, near_error=False), Entity(text='separability_matrix(cm)', label='example', start=365, end=393, source='regex', confidence=np.float64(0.11083995256824855), bm25_score=0.3333333333333333, in_code=True, near_error=False), Entity(text='separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))', label='example', start=496, end=570, source='regex', confidence=np.float64(0.10968072533280049), bm25_score=0.3333333333333333, in_code=True, near_error=False), Entity(text='separability_matrix(m.Pix2Sky_TAN() & cm)', label='example', start=918, end=964, source='regex', confidence=np.float64(0.10925725833227844), bm25_score=0.4333333333333333, in_code=True, near_error=False), Entity(text='Linear1D', label='class', start=252, end=262, source='regex', confidence=np.float64(0.10692806438808683), bm25_score=0.15, in_code=True, near_error=False), Entity(text='cm', label='variable', start=247, end=250, source='regex', confidence=np.float64(0.10432985858712608), bm25_score=0.0, in_code=True, near_error=False), Entity(text='Pix2Sky_TAN', label='class', start=520, end=533, source='regex', confidence=np.float64(0.10155976924016673), bm25_score=0.049999999999999996, in_code=True, near_error=False)]    
        try:
            # Group entities by relevance and type for better organization
            high_priority_entities = []
            code_entities = []
            
            for entity in entities[:15]:  # Focus on top 15 most relevant
                info = self._get_entity_info(entity)
                if info['confidence'] > 0.2 or info['in_code'] or info['near_error']:
                    high_priority_entities.append(info)
                elif info['type'] in ['function', 'class', 'method', 'variable']:
                    code_entities.append(info)
            
            # Display high-priority entities first
            if high_priority_entities:
                context += "### High Priority Entities\n"
                for info in high_priority_entities[:8]:
                    context += f"• **{info['name']}** ({info['type']})\n"
                    context += f"  Confidence: {info['confidence']:.3f}"
                    if info['in_code']:
                        context += " | In codebase"
                    if info['near_error']:
                        context += " | Near error location"
                    context += f"\n  BM25: {info['bm25_score']:.3f} \n\n"
            
            # Display relevant code entities
            #the relevant snippets are in /home/abenali/Enhancing-LLMs-with-Structured-Code-Data/Combining Structured Information into a Context for LLMs/outputs/all_snippets_consolidated.md
            if code_entities:
                context += "### Code Structure Elements\n"
                for info in code_entities[:5]:
                    context += f"• {info['name']} ({info['type']}) - Confidence: {info['confidence']:.3f}\n"
                context += "\n"
                
        except Exception as e:
            context += f"Error processing entities: {str(e)}\n\n"
        
        # Add structured call relationships
        #from /home/abenali/Enhancing-LLMs-with-Structured-Code-Data/Combining Structured Information into a Context for LLMs/Context_of_the_LLM/call_hierarchy.py we can imprt all_hiererchy 
        try:
            if call_hierarchy:
                context += "## Code Dependencies & Call Graph\n"
                # Process and deduplicate call relationships
                unique_calls = set()
                for item in call_hierarchy[:12]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        call_pair = (str(item[0]), str(item[1]))
                        unique_calls.add(call_pair)
                
                if unique_calls:
                    for caller, callee in sorted(unique_calls):
                        context += f"• {caller} → {callee}\n"
                    context += "\n"
        except Exception as e:
            context += f"Error processing call hierarchy: {str(e)}\n\n"
        
        # Add matched code blocks with structure
        try:
            matched_blocks = self._get_matched_blocks_for_entities(entities, max_blocks=3)
            if matched_blocks:
                context += "## Relevant Code Locations\n"
                for entity_key, blocks in list(matched_blocks.items())[:4]:
                    context += f"### {entity_key}\n"
                    for i, block in enumerate(blocks):
                        block_info = self._format_matched_block_info(block)
                        context += f"• Symbol: {block_info['symbol']}\n"
                    context += "\n"
        except Exception as e:
            context += f"Error processing matched blocks: {str(e)}\n\n"
        
        # Add concise code context
        try:
            if self.code_snippets:
                context += "## Code Context Sample\n```python\n"
                # Extract first 800 characters for balanced context
                snippet_preview = self.code_snippets[:800]
                if len(self.code_snippets) > 800:
                    snippet_preview += "\n# ... (truncated for brevity)"
                context += snippet_preview + "\n```\n\n"
        except Exception as e:
            context += f"Error adding code context: {str(e)}\n\n"
        
        context += "## Task\nAnalyze the structured information above and generate a precise patch to resolve the issue.\n"
        return context


    def _build_comprehensive_context(self, problem_statement: str, entities: List[Dict], 
                            call_hierarchy: List, structured_data: Dict) -> str:
        """
        Comprehensive context with full structured data integration.
        Maximizes information density while maintaining logical organization.
        """
        context = f"""# Comprehensive Code Analysis - Full Context

    ## Problem Statement
    {problem_statement}

    ## Complete Entity Analysis with Structural Information
    """
        
        try:
            # Organize entities by type and confidence for comprehensive view
            entity_groups = defaultdict(list)
            for i, entity in enumerate(entities):
                info = self._get_entity_info(entity)
                info['index'] = i + 1
                entity_groups[info['type']].append(info)
            
            # Display entities grouped by type
            for entity_type, entities_list in entity_groups.items():
                if entities_list:
                    context += f"\n### {entity_type.upper()} Entities\n"
                    for info in sorted(entities_list, key=lambda x: x['confidence'], reverse=True):
                        context += f"{info['index']}. **{info['name']}**\n"
                        context += f"   • Confidence: {info['confidence']:.4f} | Rank: {info['rank']}\n"
                        context += f"   • Position: {info['start']}-{info['end']} | Length: {info['length']}\n"
                        context += f"   • BM25: {info['bm25_score']:.4f} \n"
                        
                        status_flags = []
                        if info['in_code']:
                            status_flags.append("In Codebase")
                        if info['near_error']:
                            status_flags.append("Near Error")
                        if status_flags:
                            context += f"   • Status: {' | '.join(status_flags)}\n"
                        context += "\n"
        except Exception as e:
            context += f"Error in entity analysis: {str(e)}\n\n"
        
        # Complete call hierarchy with structure analysis
        try:
            if call_hierarchy:
                context += "## Complete Call Graph & Dependencies\n"
                
                # Build call graph structure
                call_graph = defaultdict(set)
                reverse_graph = defaultdict(set)
                
                for item in call_hierarchy:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        caller, callee = str(item[0]), str(item[1])
                        call_graph[caller].add(callee)
                        reverse_graph[callee].add(caller)
                
                # Display call relationships
                context += "### Function Call Relationships\n"
                for caller, callees in sorted(call_graph.items()):
                    context += f"**{caller}** calls:\n"
                    for callee in sorted(callees):
                        context += f"  → {callee}\n"
                    context += "\n"
                
                # Display dependency analysis
                context += "### Dependency Analysis\n"
                root_functions = set(call_graph.keys()) - set(reverse_graph.keys())
                leaf_functions = set(reverse_graph.keys()) - set(call_graph.keys())
                
                if root_functions:
                    context += f"Root functions (no callers): {', '.join(sorted(root_functions))}\n"
                if leaf_functions:
                    context += f"Leaf functions (no callees): {', '.join(sorted(leaf_functions))}\n"
                context += "\n"
        except Exception as e:
            context += f"Error in call hierarchy analysis: {str(e)}\n\n"
        
        # Comprehensive matched blocks analysis
        try:
            matched_blocks = self._get_matched_blocks_for_entities(entities, max_blocks=8)
            if matched_blocks:
                context += "## Detailed Code Block Analysis\n"
                for entity_key, blocks in matched_blocks.items():
                    context += f"### {entity_key}\n"
                    for i, block in enumerate(blocks):
                        block_info = self._format_matched_block_info(block)
                        context += f"**Block {i+1}**:\n"
                        context += f"  • Symbol: {block_info['symbol']}\n"
                        context += f"  • Full Path: {block_info['full_symbol']}\n"
                        if block_info['enclosing_range']:
                            context += f"  • Enclosing: {block_info['enclosing_range']}\n"
                        context += "\n"
            else:
                context += "## Code Block Analysis\nNo matched blocks found for current entities.\n\n"
        except Exception as e:
            context += f"Error in matched blocks analysis: {str(e)}\n\n"
        
        # Full code context
        try:
            if self.code_snippets:
                context += "## Complete Code Context\n"
                context += "```python\n"
                context += self.code_snippets
                context += "\n```\n\n"
        except Exception as e:
            context += f"Error adding complete code context: {str(e)}\n\n"
        
        # Additional structured data
        #metadata path: /fast/scip_workspace/astropy/formatted_output.json
        try:
            if structured_data:
                context += "## Additional Structured Information\n"
                for key, data in structured_data.items():
                    if key not in ['metadata', 'all_snippets_combined'] and data:
                        context += f"### {key.replace('_', ' ').title()}\n"
                        if isinstance(data, dict):
                            for sub_key, sub_value in list(data.items())[:5]:
                                context += f"• {sub_key}: {str(sub_value)[:100]}\n"
                        elif isinstance(data, list):
                            for i, item in enumerate(data[:5]):
                                context += f"• Item {i+1}: {str(item)[:100]}\n"
                        else:
                            context += f"{str(data)[:200]}\n"
                        context += "\n"
        except Exception as e:
            context += f"Error processing additional data: {str(e)}\n\n"
        
        context += "## Task\nUsing all the comprehensive structural information above, provide detailed analysis and generate an optimal patch solution.\n"
        return context


    def _build_rag_context(self, problem_statement: str, entities: List[Dict], 
                        call_hierarchy: List, structured_data: Dict) -> str:
        """
        RAG-style context focusing on code retrieval and semantic similarity.
        Emphasizes code snippets and their semantic relationships to the problem.
        """
        context = f"""# Code Retrieval & Generation Context

    ## Issue Description
    {problem_statement}

    ## Retrieved Code Snippets (Semantic Matching)
    """
        
        # Primary code context
        try:
            if self.code_snippets:
                context += "### Main Code Repository\n"
                context += "```python\n"
                context += self.code_snippets
                context += "\n```\n\n"
        except Exception as e:
            context += f"Error retrieving main code snippets: {str(e)}\n\n"
        
        # Semantically related code blocks
        try:
            matched_blocks = self._get_matched_blocks_for_entities(entities, max_blocks=6)
            if matched_blocks:
                context += "## Semantically Related Code Blocks\n"
                for entity_key, blocks in matched_blocks.items():
                    context += f"### Related to: {entity_key}\n"
                    for i, block in enumerate(blocks):
                        block_info = self._format_matched_block_info(block)
                        context += f"**Match {i+1}**: \n"
                        context += f"```python\n# {block_info['symbol']}\n ```\n\n"
        except Exception as e:
            context += f"Error processing semantic matches: {str(e)}\n\n"
        
        # Entity-based code retrieval
        try:
            context += "## Key Symbols & Code Elements\n"
            
            # Group entities by confidence for retrieval ranking
            high_conf = [e for e in entities if self._get_entity_info(e)['confidence'] > 0.3]
            medium_conf = [e for e in entities if 0.1 < self._get_entity_info(e)['confidence'] <= 0.3]
            
            if high_conf:
                context += "### High Confidence Matches\n"
                for entity in high_conf[:8]:
                    info = self._get_entity_info(entity)
                    context += f"• **{info['name']}** ({info['type']}) - Score: {info['confidence']:.3f}\n"
                    if info['bm25_score'] > 0:
                        context += f"  BM25 Relevance: {info['bm25_score']:.3f}\n"
                context += "\n"
            
            if medium_conf:
                context += "### Medium Confidence Matches\n"
                for entity in medium_conf[:6]:
                    info = self._get_entity_info(entity)
                    context += f"• {info['name']} ({info['type']}) - Score: {info['confidence']:.3f}\n"
                context += "\n"
        except Exception as e:
            context += f"Error processing entity-based retrieval: {str(e)}\n\n"
        
        # Function call context for RAG
        try:
            if call_hierarchy:
                context += "## Function Call Context\n"
                context += "```\n# Call relationships (for context):\n"
                
                for item in call_hierarchy[:10]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        context += f"# {item[0]} → {item[1]}\n"
                context += "```\n\n"
        except Exception as e:
            context += f"Error adding call context: {str(e)}\n\n"
        
        # Additional retrieved context
        try:
            if structured_data.get('all_snippets_combined'):
                context += "## Additional Retrieved Context\n"
                additional_snippets = structured_data['all_snippets_combined']
                if isinstance(additional_snippets, str) and len(additional_snippets) > 0:
                    context += "```python\n"
                    context += additional_snippets[:1000]  # Limit for RAG context
                    if len(additional_snippets) > 1000:
                        context += "\n# ... (additional snippets available)"
                    context += "\n```\n\n"
        except Exception as e:
            context += f"Error adding additional context: {str(e)}\n\n"
        
        context += "## Generation Task\nBased on the retrieved code snippets and semantic matches above, generate a targeted patch that addresses the specific issue.\n"
        return context


    def _build_hierarchical_context(self, problem_statement: str, entities: List[Dict], 
                                call_hierarchy: List, structured_data: Dict) -> str:
        """
        Hierarchical context organized by code structure and architectural layers.
        Presents information in a top-down, structured manner following code organization principles.
        """
        context = f"""# Hierarchical Code Structure Analysis

    ## Problem Statement
    {problem_statement}

    ## Code Architecture Overview
    """
        
        # Layer 1: High-level entity classification
        try:
            # Classify entities into architectural layers
            layers = {
                'interfaces': [],
                'classes': [],
                'functions': [],
                'variables': [],
                'modules': [],
                'other': []
            }
            
            for entity in entities:
                info = self._get_entity_info(entity)
                entity_type = info['type'].lower()
                
                if entity_type in ['interface', 'protocol']:
                    layers['interfaces'].append(info)
                elif entity_type in ['class', 'struct']:
                    layers['classes'].append(info)
                elif entity_type in ['function', 'method', 'procedure']:
                    layers['functions'].append(info)
                elif entity_type in ['variable', 'field', 'property']:
                    layers['variables'].append(info)
                elif entity_type in ['module', 'package', 'namespace']:
                    layers['modules'].append(info)
                else:
                    layers['other'].append(info)
            
            # Display architectural layers
            layer_names = {
                'modules': 'Module/Package Layer',
                'interfaces': 'Interface/Protocol Layer', 
                'classes': 'Class/Type Layer',
                'functions': 'Function/Method Layer',
                'variables': 'Variable/Field Layer',
                'other': 'Other Elements'
            }
            
            for layer_key, layer_name in layer_names.items():
                entities_in_layer = layers[layer_key]
                if entities_in_layer:
                    context += f"\n### {layer_name}\n"
                    # Sort by confidence within each layer
                    sorted_entities = sorted(entities_in_layer, key=lambda x: x['confidence'], reverse=True)
                    for info in sorted_entities[:8]:  # Limit per layer
                        context += f"• **{info['name']}**\n"
                        context += f"  Confidence: {info['confidence']:.3f} | Rank: {info['rank']}\n"
                        if info['in_code'] or info['near_error']:
                            status = []
                            if info['in_code']:
                                status.append("In Code")
                            if info['near_error']:
                                status.append("Near Error")
                            context += f"  Status: {' | '.join(status)}\n"
                        context += f"  BM25: {info['bm25_score']:.3f}\n\n"
                        
        except Exception as e:
            context += f"Error in architectural analysis: {str(e)}\n\n"
        
        # Layer 2: Call hierarchy organization
        try:
            if call_hierarchy:
                context += "## Function Call Hierarchy\n"
                
                # Build hierarchical call structure
                call_tree = defaultdict(list)
                all_callees = set()
                
                for item in call_hierarchy:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        caller, callee = str(item[0]), str(item[1])
                        call_tree[caller].append(callee)
                        all_callees.add(callee)
                
                # Find root functions (functions that call others but aren't called)
                root_functions = set(call_tree.keys()) - all_callees
                
                # Display hierarchical structure
                def display_call_tree(func, level=0, visited=None):
                    if visited is None:
                        visited = set()
                    if func in visited or level > 3:  # Prevent infinite recursion
                        return ""
                    
                    visited.add(func)
                    indent = "  " * level
                    result = f"{indent}• {func}\n"
                    
                    for callee in call_tree.get(func, []):
                        result += display_call_tree(callee, level + 1, visited.copy())
                    
                    return result
                
                if root_functions:
                    context += "### Call Trees (Top-Down)\n"
                    for root in sorted(root_functions):
                        context += display_call_tree(root)
                    context += "\n"
                else:
                    context += "### Function Relationships\n"
                    for caller, callees in sorted(call_tree.items()):
                        context += f"• **{caller}** calls:\n"
                        for callee in callees:
                            context += f"  → {callee}\n"
                    context += "\n"
                    
        except Exception as e:
            context += f"Error in call hierarchy analysis: {str(e)}\n\n"
        
        # Layer 3: Code structure with matched blocks
        try:
            matched_blocks = self._get_matched_blocks_for_entities(entities, max_blocks=5)
            if matched_blocks:
                context += "## Code Structure Mapping\n"
                
                # Organize blocks by location/file structure
                location_groups = defaultdict(list)
                for entity_key, blocks in matched_blocks.items():
                    for block in blocks:
                        block_info = self._format_matched_block_info(block)
                        location_groups[entity_key].append(block_info)
                
                for entity_key, block_infos in location_groups.items():
                    context += f"### {entity_key}\n"
                    for i, block_info in enumerate(block_infos):
                        context += f"  • Symbol: {block_info['symbol']}\n\n"
        except Exception as e:
            context += f"Error in code structure mapping: {str(e)}\n\n"
        
        # Layer 4: Implementation details
        try:
            if self.code_snippets:
                context += "## Implementation Layer\n"
                context += "### Core Code Implementation\n"
                context += "```python\n"
                # For hierarchical context, show code in structured way
                lines = self.code_snippets.split('\n')
                if len(lines) > 30:
                    # Show beginning and end for hierarchical overview
                    context += '\n'.join(lines[:15])
                    context += f"\n\n# ... ({len(lines) - 30} lines of implementation) ...\n\n"
                    context += '\n'.join(lines[-15:])
                else:
                    context += self.code_snippets
                context += "\n```\n\n"
        except Exception as e:
            context += f"Error in implementation layer: {str(e)}\n\n"
        
        # Layer 5: High-confidence elements summary
        try:
            high_confidence_entities = [
                e for e in entities 
                if self._get_entity_info(e)['confidence'] > 0.2
            ]
            
            if high_confidence_entities:
                context += "## High-Confidence Analysis Summary\n"
                context += "### Critical Elements (Confidence > 0.2)\n"
                for entity in high_confidence_entities[:8]:
                    info = self._get_entity_info(entity)
                    context += f"• **{info['name']}** ({info['type']}) - {info['confidence']:.3f}\n"
                    if info['near_error']:
                        context += f"  ⚠ Located near error context\n"
                context += "\n"
        except Exception as e:
            context += f"Error in confidence analysis: {str(e)}\n\n"
        
        context += "## Synthesis Task\nAnalyze the hierarchical code structure above and generate a well-structured patch that respects the architectural organization.\n"
        return context