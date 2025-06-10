import json
import os
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging
from collections import defaultdict

from config import MODEL_CONFIGS, STRATEGIES, problem_stmt
import sys
import os
from call_hierarchy import analyzer_instance, all_hierarchy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ranking_entities import (
    entities, updated_entities, weights, code_positions,
    error_positions, auto_query
)

from llm_models import create_model

logger = logging.getLogger(__name__)

def load_matched_blocks(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
        
def load_code_snippets(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_entity_attr(entity, attr_name: str, default: str = 'unknown'):
    """Safely get entity attribute whether it's a dict or object"""
    if hasattr(entity, attr_name):
        return getattr(entity, attr_name)
    elif isinstance(entity, dict) and attr_name in entity:
        return entity[attr_name]
    else:
        return default

class ContextStrategy:
    """Context generation strategies for different approaches"""
    
    @staticmethod
    def _get_patch_instructions() -> str:
        """Common patch generation instructions"""
        return """
## PATCH GENERATION INSTRUCTIONS

You are a software engineer tasked with fixing code issues. Follow these steps:

1. **ANALYZE THE PROBLEM**: Carefully read and understand the problem statement
2. **EXAMINE THE CODE**: Study the provided code context and identify the root cause
3. **PLAN THE SOLUTION**: Determine what changes are needed to fix the issue
4. **GENERATE THE PATCH**: Create a unified diff patch in the exact format below

## REQUIRED OUTPUT FORMAT

Your response MUST follow this exact structure:

### ANALYSIS
[Explain what the problem is and why it occurs]

### SOLUTION
[Describe your fix approach and reasoning]

### PATCH
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line_start,line_count +line_start,line_count @@
 context_line
-removed_line
+added_line
 context_line
```

## PATCH FORMAT RULES
- Use unified diff format (--- +++ @@ syntax)
- Include file paths relative to project root
- Show 3 lines of context before and after changes
- Use '-' for removed lines, '+' for added lines
- Include line numbers in @@ headers
- Make minimal, targeted changes only

## IMPORTANT REQUIREMENTS
- Generate EXACTLY ONE patch per file that needs changes
- Only modify lines that are directly related to the problem
- Preserve all existing functionality not related to the bug
- Use the exact indentation and formatting from the original code
- Test your logic mentally before outputting the patch
"""

    @staticmethod
    def minimal(problem_statement: str, entities: List[Any], code_snippets: str) -> str:
        """Minimal context with only essential information"""
        context = f"""# CODE REPAIR TASK

## PROBLEM STATEMENT
{problem_statement}

## KEY ENTITIES (Top 10 Most Relevant)
"""
        for i, entity in enumerate(entities[:10], 1):
            name = get_entity_attr(entity, 'name')
            entity_type = get_entity_attr(entity, 'type')
            file_path = get_entity_attr(entity, 'file')
            score = get_entity_attr(entity, 'score', 0)
            context += f"{i}. **{name}** ({entity_type}) in `{file_path}` [Score: {score}]\n"
        
        context += f"""
## CRITICAL CODE CONTEXT
```python
{code_snippets[:1500]}
```

{ContextStrategy._get_patch_instructions()}

**TASK**: Fix the issue described in the problem statement using the provided code context. Generate a precise patch that addresses the root cause.
"""
        return context
    
    @staticmethod
    def balanced(problem_statement: str, entities: List[Any], code_snippets: str, 
                matched_blocks: Dict) -> str:
        """Balanced context with moderate detail"""
        context = f"""# COMPREHENSIVE CODE REPAIR TASK

## PROBLEM STATEMENT
{problem_statement}

## RELEVANT ENTITIES AND THEIR CONTEXT
"""
        for i, entity in enumerate(entities[:8], 1):
            name = get_entity_attr(entity, 'name')
            entity_type = get_entity_attr(entity, 'type')
            file_path = get_entity_attr(entity, 'file')
            score = get_entity_attr(entity, 'score', 0)
            description = get_entity_attr(entity, 'description', '')
            
            context += f"""### {i}. {name} ({entity_type})
- **File**: `{file_path}`
- **Relevance Score**: {score}
- **Description**: {description}

"""
        
        context += f"""## PRIMARY CODE CONTEXT
```python
{code_snippets[:2500]}
```
"""
        
        if 'matched_blocks' in matched_blocks:
            context += "\n## ADDITIONAL RELEVANT CODE BLOCKS\n"
            for i, block in enumerate(matched_blocks['matched_blocks'][:4], 1):
                func_name = block.get('function', 'Unknown')
                relevance = block.get('relevance_score', 0)
                code = block.get('code', '')
                context += f"""### Block {i}: {func_name} (Relevance: {relevance:.3f})
```python
{code}
```

"""
        
        context += f"""
{ContextStrategy._get_patch_instructions()}

**TASK**: 
1. Analyze the problem statement in relation to the provided code
2. Identify which specific files and functions need to be modified
3. Generate appropriate patches to fix the issue
4. Ensure your solution addresses the root cause, not just symptoms
"""
        return context
    
    @staticmethod
    def comprehensive(problem_statement: str, entities: List[Any], code_snippets: str,
                     matched_blocks: Dict, call_hierarchy: List = None) -> str:
        """Comprehensive context with full detail"""
        context = f"""# DETAILED CODE ANALYSIS AND REPAIR TASK

## PROBLEM STATEMENT
{problem_statement}

## COMPLETE ENTITY ANALYSIS
"""
        for i, entity in enumerate(entities, 1):
            name = get_entity_attr(entity, 'name')
            entity_type = get_entity_attr(entity, 'type')
            file_path = get_entity_attr(entity, 'file')
            score = get_entity_attr(entity, 'score', 0)
            description = get_entity_attr(entity, 'description', '')
            
            context += f"""### {i}. {name}
- **Type**: {entity_type}
- **File**: `{file_path}`
- **Relevance Score**: {score}
- **Description**: {description}

"""
        
        context += f"""## COMPLETE CODE CONTEXT
```python
{code_snippets}
```
"""
        
        if 'matched_blocks' in matched_blocks:
            context += "\n## ALL MATCHED CODE BLOCKS\n"
            for i, block in enumerate(matched_blocks['matched_blocks'], 1):
                func_name = block.get('function', 'Unknown')
                relevance = block.get('relevance_score', 0)
                code = block.get('code', '')
                file_info = block.get('file', 'Unknown file')
                context += f"""### Block {i}: {func_name} (Score: {relevance:.3f})
**File**: `{file_info}`
```python
{code}
```

"""
        
        if call_hierarchy:
            context += "\n## CALL HIERARCHY ANALYSIS\n"
            context += "Understanding function call relationships:\n"
            for caller, callee in call_hierarchy:
                context += f"- `{caller}` calls `{callee}`\n"
            context += "\n"
        
        context += f"""
{ContextStrategy._get_patch_instructions()}

**COMPREHENSIVE TASK**:
1. **Deep Analysis**: Thoroughly understand the problem and its context
2. **Root Cause Identification**: Use all provided information to identify the exact cause
3. **Impact Assessment**: Consider how your changes affect the call hierarchy and related code
4. **Solution Design**: Plan a complete solution that addresses all aspects of the problem
5. **Patch Generation**: Create precise patches with proper diff format
6. **Validation**: Mentally verify your solution handles edge cases

**FOCUS AREAS**:
- Use the entity rankings to prioritize which code sections are most relevant
- Consider the call hierarchy when making changes to avoid breaking dependencies
- Pay attention to the matched code blocks for additional context
- Make minimal but complete changes
"""
        return context
    
    @staticmethod
    def rag_style(problem_statement: str, entities: List[Any], code_snippets: str,
                  matched_blocks: Dict) -> str:
        """RAG-style context focusing on code retrieval"""
        context = f"""# RETRIEVAL-AUGMENTED CODE REPAIR

## TARGET ISSUE
{problem_statement}

## RETRIEVED CODE SNIPPETS (Ranked by Relevance)

### Primary Code Context
```python
{code_snippets}
```

"""
        
        if 'matched_blocks' in matched_blocks:
            context += "### Additional Retrieved Code Snippets\n"
            for i, block in enumerate(matched_blocks['matched_blocks'][:6], 1):
                func_name = block.get('function', 'Unknown')
                relevance = block.get('relevance_score', 0)
                code = block.get('code', '')
                file_info = block.get('file', 'Unknown')
                
                context += f"""#### Snippet {i}: `{func_name}` (Relevance: {relevance:.3f})
**Source**: `{file_info}`
```python
{code}
```

"""
        
        context += "## KEY SYMBOLS AND ENTITIES\n"
        for i, entity in enumerate(entities[:8], 1):
            name = get_entity_attr(entity, 'name')
            entity_type = get_entity_attr(entity, 'type')
            file_path = get_entity_attr(entity, 'file')
            context += f"{i}. **{name}** ({entity_type}) - `{file_path}`\n"
        
        context += f"""

{ContextStrategy._get_patch_instructions()}

**RAG-BASED REPAIR TASK**:
1. **Information Retrieval**: Use the ranked code snippets as your primary information source
2. **Relevance Assessment**: Focus on the highest-scoring snippets and entities
3. **Pattern Matching**: Look for patterns in the retrieved code that relate to the problem
4. **Targeted Fixing**: Generate patches based on the most relevant retrieved information
5. **Context Preservation**: Ensure changes are consistent with the broader codebase patterns

**RETRIEVAL FOCUS**: The code snippets have been specifically retrieved and ranked for this problem. Use them as your primary source of truth for understanding the codebase structure and implementing the fix.
"""
        return context

class SWEPatchGenerator:
    """Main patch generator class"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        self.models = {}
        self.results = defaultdict(dict)
        self.metrics = []
        
        # Use absolute path instead of relative path
        json_file_path = r"C:\Users\Asus\Desktop\Enhancing-LLMs-with-Structured-Code-Data\Combining Structured Information into a Context for LLMs\matched_blocks_ranked.json"
        self.matched_blocks = load_matched_blocks(json_file_path)
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_script_dir))  
        self.data_dir = project_root
        code_snippets_path = os.path.join(self.data_dir, "outputs", "all_snippets_consolidated.py")
        self.code_snippets = load_code_snippets(code_snippets_path)
        
        # Create call hierarchy
        self.call_hierarchy = all_hierarchy
        
        logger.info("SWE Patch Generator initialized")
    
    def initialize_models(self):
        """Initialize all LLM models"""
        logger.info("Initializing models...")
        
        for model_key, config in MODEL_CONFIGS.items():
            try:
                logger.info(f"Loading {model_key}...")
                model = create_model(config['name'], config)
                model.load_model()
                self.models[model_key] = model
                logger.info(f"✓ {model_key} loaded successfully")
            except Exception as e:
                logger.error(f"✗ Failed to load {model_key}: {str(e)}")
                # Continue with other models
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def generate_contexts(self) -> Dict[str, str]:
        """Generate contexts for all strategies"""
        contexts = {}
        
        # Minimal strategy
        contexts['minimal'] = ContextStrategy.minimal(
            problem_stmt, updated_entities, self.code_snippets
        )
        
        # Balanced strategy
        contexts['balanced'] = ContextStrategy.balanced(
            problem_stmt, updated_entities, self.code_snippets, self.matched_blocks
        )
        
        # Comprehensive strategy
        contexts['comprehensive'] = ContextStrategy.comprehensive(
            problem_stmt, updated_entities, self.code_snippets, 
            self.matched_blocks, self.call_hierarchy
        )
        
        # RAG-style strategy
        contexts['rag_style'] = ContextStrategy.rag_style(
            problem_stmt, updated_entities, self.code_snippets, self.matched_blocks
        )
        
        return contexts
    
    def validate_patch_format(self, patch_text: str) -> Tuple[bool, str]:
        """Validate that the generated patch follows correct format"""
        if not patch_text or not patch_text.strip():
            return False, "Empty patch generated"
        
        # Check for diff markers
        has_diff_header = any(line.startswith('---') or line.startswith('+++') for line in patch_text.split('\n'))
        has_hunk_header = '@@' in patch_text
        has_changes = any(line.startswith('+') or line.startswith('-') for line in patch_text.split('\n'))
        
        if not has_diff_header:
            return False, "Missing diff header (--- +++)"
        if not has_hunk_header:
            return False, "Missing hunk header (@@)"
        if not has_changes:
            return False, "No actual changes found (+ or - lines)"
        
        return True, "Valid patch format"
    
    def extract_patch_from_response(self, response: str) -> str:
        """Extract patch content from LLM response"""
        lines = response.split('\n')
        patch_lines = []
        in_patch = False
        
        for line in lines:
            # Look for patch section
            if line.strip().upper() in ['### PATCH', '## PATCH', 'PATCH:', '```diff']:
                in_patch = True
                continue
            elif line.strip() == '```' and in_patch:
                break
            elif in_patch and (line.startswith('---') or line.startswith('+++') or 
                             line.startswith('@@') or line.startswith(' ') or 
                             line.startswith('+') or line.startswith('-')):
                patch_lines.append(line)
        
        return '\n'.join(patch_lines)
    
    def generate_all_patches(self) -> Dict[str, Dict[str, Dict]]:
        """Generate patches for all model-strategy combinations"""
        logger.info("Starting patch generation for all combinations...")
        
        contexts = self.generate_contexts()
        results = defaultdict(dict)
        
        total_combinations = len(self.models) * len(STRATEGIES)
        current = 0
        
        for model_name, model in self.models.items():
            for strategy in STRATEGIES:
                current += 1
                logger.info(f"[{current}/{total_combinations}] Generating: {model_name} + {strategy}")
                
                try:
                    context = contexts[strategy]
                    response, metrics = model.generate_patch(context, strategy)
                    
                    # Extract and validate patch
                    extracted_patch = self.extract_patch_from_response(response)
                    is_valid, validation_msg = self.validate_patch_format(extracted_patch)
                    
                    # Update metrics with validation info
                    metrics.update({
                        'patch_extracted': bool(extracted_patch),
                        'patch_valid': is_valid,
                        'validation_message': validation_msg,
                        'extracted_patch_length': len(extracted_patch)
                    })
                    
                    results[model_name][strategy] = {
                        'response': response,
                        'extracted_patch': extracted_patch,
                        'metrics': metrics,
                        'context': context,
                        'timestamp': datetime.now().isoformat(),
                        'validation': {
                            'is_valid': is_valid,
                            'message': validation_msg
                        }
                    }
                    
                    self.metrics.append(metrics)
                    
                    if is_valid:
                        logger.info(f"✓ Generated valid patch ({len(extracted_patch)} chars)")
                    else:
                        logger.warning(f"⚠ Generated response but patch invalid: {validation_msg}")
                    
                except Exception as e:
                    logger.error(f"✗ Generation failed: {str(e)}")
                    results[model_name][strategy] = {
                        'response': f"Error: {str(e)}",
                        'extracted_patch': '',
                        'metrics': {'error': str(e), 'success': False},
                        'context': contexts.get(strategy, ''),
                        'timestamp': datetime.now().isoformat(),
                        'validation': {
                            'is_valid': False,
                            'message': f"Generation error: {str(e)}"
                        }
                    }
        
        self.results = results
        logger.info("Patch generation completed")
        return results
    
    def save_results(self, output_dir: str = "results"):
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual results
        for model_name, model_results in self.results.items():
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            for strategy, result in model_results.items():
                strategy_dir = os.path.join(model_dir, strategy)
                os.makedirs(strategy_dir, exist_ok=True)
                
                # Save full response
                with open(os.path.join(strategy_dir, "full_response.txt"), 'w', encoding='utf-8') as f:
                    f.write(result['response'])
                
                # Save extracted patch
                with open(os.path.join(strategy_dir, "patch.diff"), 'w', encoding='utf-8') as f:
                    f.write(result.get('extracted_patch', ''))
                
                # Save context
                with open(os.path.join(strategy_dir, "context.txt"), 'w', encoding='utf-8') as f:
                    f.write(result['context'])
                
                # Save metrics and validation
                with open(os.path.join(strategy_dir, "metrics.json"), 'w', encoding='utf-8') as f:
                    json.dump(result['metrics'], f, indent=2, default=str)
                
                # Save validation results
                with open(os.path.join(strategy_dir, "validation.json"), 'w', encoding='utf-8') as f:
                    json.dump(result['validation'], f, indent=2)
        
        # Save summary
        with open(os.path.join(output_dir, "summary.json"), 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save metrics
        with open(os.path.join(output_dir, "all_metrics.json"), 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_dir}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all models and strategies"""
        if not self.metrics:
            return {"message": "No metrics available"}
        
        # Count successful patch generations
        valid_patches = sum(1 for m in self.metrics if m.get('patch_valid', False))
        extracted_patches = sum(1 for m in self.metrics if m.get('patch_extracted', False))
        
        summary = {
            "total_generations": len(self.metrics),
            "successful_generations": sum(1 for m in self.metrics if m.get('success', False)),
            "valid_patches_generated": valid_patches,
            "patches_extracted": extracted_patches,
            "patch_success_rate": valid_patches / len(self.metrics) * 100 if self.metrics else 0,
            "average_generation_time": sum(m.get('generation_time', 0) for m in self.metrics) / len(self.metrics),
            "average_tokens_per_second": sum(m.get('tokens_per_second', 0) for m in self.metrics) / len(self.metrics),
            "model_performance": defaultdict(dict),
            "strategy_performance": defaultdict(dict)
        }
        
        # Group by model and strategy
        for metric in self.metrics:
            model = metric.get('model', 'unknown')
            strategy = metric.get('strategy', 'unknown')
            
            # Initialize model metrics
            if model not in summary["model_performance"]:
                summary["model_performance"][model] = {
                    "generations": 0,
                    "valid_patches": 0,
                    "avg_time": 0,
                    "avg_tokens_per_sec": 0,
                    "success_rate": 0
                }
            
            # Initialize strategy metrics
            if strategy not in summary["strategy_performance"]:
                summary["strategy_performance"][strategy] = {
                    "generations": 0,
                    "valid_patches": 0,
                    "avg_context_length": 0,
                    "avg_output_tokens": 0
                }
            
            # Update model metrics
            model_perf = summary["model_performance"][model]
            model_perf["generations"] += 1
            model_perf["valid_patches"] += 1 if metric.get('patch_valid', False) else 0
            model_perf["avg_time"] += metric.get('generation_time', 0)
            model_perf["avg_tokens_per_sec"] += metric.get('tokens_per_second', 0)
            model_perf["success_rate"] += 1 if metric.get('success', False) else 0
            
            # Update strategy metrics
            strategy_perf = summary["strategy_performance"][strategy]
            strategy_perf["generations"] += 1
            strategy_perf["valid_patches"] += 1 if metric.get('patch_valid', False) else 0
            strategy_perf["avg_context_length"] += metric.get('context_length', 0)
            strategy_perf["avg_output_tokens"] += metric.get('output_tokens', 0)
        
        # Calculate averages and percentages
        for model_data in summary["model_performance"].values():
            if model_data["generations"] > 0:
                model_data["avg_time"] /= model_data["generations"]
                model_data["avg_tokens_per_sec"] /= model_data["generations"]
                model_data["success_rate"] = (model_data["success_rate"] / model_data["generations"]) * 100
                model_data["patch_success_rate"] = (model_data["valid_patches"] / model_data["generations"]) * 100
        
        for strategy_data in summary["strategy_performance"].values():
            if strategy_data["generations"] > 0:
                strategy_data["avg_context_length"] /= strategy_data["generations"]
                strategy_data["avg_output_tokens"] /= strategy_data["generations"]
                strategy_data["patch_success_rate"] = (strategy_data["valid_patches"] / strategy_data["generations"]) * 100
        
        return summary

    def print_performance_report(self):
        """Print a detailed performance report"""
        summary = self.get_performance_summary()
        
        print("="*60)
        print("SWE PATCH GENERATOR PERFORMANCE REPORT")
        print("="*60)
        
        print(f"Total Generations: {summary['total_generations']}")
        print(f"Valid Patches Generated: {summary['valid_patches_generated']}")
        print(f"Overall Patch Success Rate: {summary['patch_success_rate']:.1f}%")
        print(f"Average Generation Time: {summary['average_generation_time']:.2f}s")
        
        print("\n" + "-"*40)
        print("MODEL PERFORMANCE")
        print("-"*40)
        for model, perf in summary['model_performance'].items():
            print(f"\n{model}:")
            print(f"  Generations: {perf['generations']}")
            print(f"  Valid Patches: {perf['valid_patches']}")
            print(f"  Patch Success Rate: {perf.get('patch_success_rate', 0):.1f}%")
            print(f"  Avg Generation Time: {perf['avg_time']:.2f}s")
        
        print("\n" + "-"*40)
        print("STRATEGY PERFORMANCE")
        print("-"*40)
        for strategy, perf in summary['strategy_performance'].items():
            print(f"\n{strategy}:")
            print(f"  Generations: {perf['generations']}")
            print(f"  Valid Patches: {perf['valid_patches']}")
            print(f"  Patch Success Rate: {perf.get('patch_success_rate', 0):.1f}%")
            print(f"  Avg Context Length: {perf['avg_context_length']:.0f} chars")


# Example usage
if __name__ == "__main__":
    # Initialize the patch generator
    generator = SWEPatchGenerator()
    
    # Initialize models
    generator.initialize_models()
    
    # Generate all patches
    results = generator.generate_all_patches()
    
    # Save results
    generator.save_results()
    
    # Print performance report
    generator.print_performance_report()
    
    print("\nPatch generation completed! Check the 'results' directory for output files.")