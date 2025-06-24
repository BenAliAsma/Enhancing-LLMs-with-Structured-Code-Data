import os
import sys
import json
import time
import traceback
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from contextlib import contextmanager
import threading

import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Import your existing modules
from src.config import MODEL_CONFIGS, STRATEGIES, problem_stmt
from call_hierarchy import analyzer_instance, all_hierarchy

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ranking_entities import (
    entities, updated_entities, weights, code_positions,
    error_positions, auto_query, _process_entities
)
# Updated import to use the new quantized model system
from src.llm_models import create_quantized_model, QuantizedHuggingFaceLLM

# Import our new modules
from src.context_builder import ContextBuilder
from src.results_analyzer import ResultsAnalyzer
from src.utils import setup_logging, create_output_dir
from src.patch_generator import PatchGenerator


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


@contextmanager
def timeout_context(seconds: int, operation_name: str = "operation"):
    """
    Context manager for timing out operations
    
    Args:
        seconds: Timeout in seconds
        operation_name: Name of the operation for error messages
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Timeout: {operation_name} exceeded {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def progress_tracker(current: int, total: int, operation: str, start_time: float) -> None:
    """
    Print progress information with timing estimates
    
    Args:
        current: Current progress count
        total: Total items to process
        operation: Description of the operation
        start_time: Start time of the operation
    """
    elapsed = time.time() - start_time
    if current > 0:
        avg_time = elapsed / current
        remaining = total - current
        eta = avg_time * remaining
        eta_str = f"ETA: {eta/60:.1f}m" if eta > 60 else f"ETA: {eta:.1f}s"
    else:
        eta_str = "ETA: calculating..."
    
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"\r{operation}: {current}/{total} ({percentage:.1f}%) - {eta_str}", end="", flush=True)


def convert_entities_to_dicts(entities_list: List, problem_stmt: str, weights: Dict) -> List[Dict]:
    """
    Convert Entity objects to dictionaries that the context builder expects
    
    Args:
        entities_list: List of entity objects
        problem_stmt: Problem statement string
        weights: Dictionary of weights for different entity types
        
    Returns:
        List of entity dictionaries
    """
    print("Converting entities to dictionaries...")
    try:
        # Add timeout for entity processing
        with timeout_context(300, "entity processing"):  # 5 minute timeout
            processed_entities = _process_entities(entities_list, problem_stmt, weights)
        print(f"\n✓ Successfully processed {len(processed_entities)} entities")
        return processed_entities
    except TimeoutError as e:
        print(f"\n✗ Entity processing timed out: {str(e)}")
        return []
    except Exception as e:
        print(f"\n✗ Error processing entities: {str(e)}")
        traceback.print_exc()
        return []


def validate_dependencies() -> bool:
    """
    Validate that all required dependencies and files are available
    
    Returns:
        True if all dependencies are satisfied, False otherwise
    """
    print("Validating dependencies...")
    required_modules = [
        'src.config', 'src.llm_models', 'src.context_builder',
        'src.results_analyzer', 'src.utils', 'src.patch_generator'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError as e:
            missing_modules.append(f"{module}: {str(e)}")
    
    if missing_modules:
        print("Missing required modules:")
        for module in missing_modules:
            print(f"  - {module}")
        return False
    
    # Check for required variables
    required_vars = {
        "MODEL_CONFIGS": MODEL_CONFIGS,
        "STRATEGIES": STRATEGIES,
        "problem_stmt": problem_stmt,
        "updated_entities": updated_entities,
        "weights": weights
    }
    missing_vars = [name for name, val in required_vars.items() if val is None]
    if missing_vars:
        print(f"Missing required variables: {', '.join(missing_vars)}")
        return False
    
    print("✓ All dependencies validated")
    return True


def load_structured_data() -> Dict[str, Any]:
    """
    Load structured data from existing files with error handling and timeout
    
    Returns:
        Dictionary containing structured data
    """
    print("Loading structured data...")
    structured_data = {}
    
    # Define data files to load with correct file paths
    data_files = {
        "/fast/scip_workspace/astropy/formatted_snapshot.json": "metadata",
        "/home/abenali/Enhancing-LLMs-with-Structured-Code-Data/Combining Structured Information into a Context for LLMs/matched_blocks_ranked.json": "matched_blocks",
        "/home/abenali/Enhancing-LLMs-with-Structured-Code-Data/Combining Structured Information into a Context for LLMs/outputs/all_snippets_consolidated.md": "all_snippets_combined"
    }
    
    for i, (file_path, key) in enumerate(data_files.items(), 1):
        progress_tracker(i-1, len(data_files), "Loading files", time.time())
        
        if os.path.exists(file_path):
            try:
                # Add timeout for file loading
                with timeout_context(60, f"loading {file_path}"):  # 1 minute per file
                    file_size = os.path.getsize(file_path)
                    if file_size > 100 * 1024 * 1024:  # 100MB
                        print(f"\n⚠ Large file detected: {file_path} ({file_size/1024/1024:.1f}MB)")
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if file_path.endswith('.json'):
                            structured_data[key] = json.load(f)
                        elif file_path.endswith('.md'):
                            structured_data[key] = f.read()  # Read markdown as text
                    
            except TimeoutError as e:
                print(f"\n⚠ Timeout loading {file_path}: {str(e)}")
                structured_data[key] = {} if file_path.endswith('.json') else ""
            except (json.JSONDecodeError, IOError) as e:
                print(f"\n⚠ Failed to load {file_path}: {str(e)}")
                structured_data[key] = {} if file_path.endswith('.json') else ""
        else:
            print(f"\n⚠ File not found: {file_path}")
            structured_data[key] = {} if file_path.endswith('.json') else ""
    
    progress_tracker(len(data_files), len(data_files), "Loading files", time.time())
    print(f"\n✓ Loaded structured data with {len([k for k, v in structured_data.items() if v])} valid components")
    return structured_data


def cleanup_gpu_memory():
    """Clean up GPU memory if available"""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"Warning: GPU cleanup failed: {str(e)}")


def save_results(results: Dict, analysis: Dict, contexts: Dict, output_dir: str) -> None:
    """
    Save all results to files with comprehensive error handling and progress tracking
    
    Args:
        results: Dictionary of results from all model-strategy combinations
        analysis: Analysis results
        contexts: Generated contexts for each strategy
        output_dir: Output directory path
    """
    print("Saving results...")
    start_time = time.time()
    
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
        save_operations = [
            ("results.json", lambda: json.dump(results, open(os.path.join(output_dir, "results.json"), 'w', encoding='utf-8'), indent=2, default=str, ensure_ascii=False)),
            ("analysis.json", lambda: json.dump(analysis, open(os.path.join(output_dir, "analysis.json"), 'w', encoding='utf-8'), indent=2, default=str, ensure_ascii=False)),
            ("contexts", lambda: save_contexts(contexts, output_dir)),
            ("patches", lambda: save_patches(results, output_dir)),
            ("summary.json", lambda: save_summary(results, output_dir))
        ]
        
        for i, (name, operation) in enumerate(save_operations, 1):
            progress_tracker(i-1, len(save_operations), "Saving", start_time)
            try:
                with timeout_context(120, f"saving {name}"):  # 2 minute timeout per operation
                    operation()
            except TimeoutError as e:
                print(f"\n⚠ Timeout saving {name}: {str(e)}")
            except Exception as e:
                print(f"\n⚠ Error saving {name}: {str(e)}")
        
        progress_tracker(len(save_operations), len(save_operations), "Saving", start_time)
        print(f"\n✓ Results saved to {output_dir}")
        
    except Exception as e:
        print(f"\n✗ Error saving results: {str(e)}")
        traceback.print_exc()


def save_contexts(contexts: Dict, output_dir: str) -> None:
    """Save contexts to individual files"""
    contexts_dir = os.path.join(output_dir, "contexts")
    os.makedirs(contexts_dir, exist_ok=True)
    
    for strategy, context in contexts.items():
        context_file = os.path.join(contexts_dir, f"{strategy}.txt")
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write(str(context))


def save_patches(results: Dict, output_dir: str) -> int:
    """Save individual patches and return count"""
    patches_dir = os.path.join(output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)
    
    patch_count = 0
    for model_name, model_results in results.items():
        model_dir = os.path.join(patches_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        for strategy, result in model_results.items():
            if result.get('success') and 'patch' in result:
                patch_file = os.path.join(model_dir, f"{strategy}.patch")
                with open(patch_file, 'w', encoding='utf-8') as f:
                    f.write(result['patch'])
                patch_count += 1
    
    return patch_count


def save_summary(results: Dict, output_dir: str) -> None:
    """Save execution summary"""
    patch_count = save_patches(results, output_dir)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_models': len(results),
        'total_strategies': len(STRATEGIES),
        'successful_patches': patch_count,
        'output_directory': output_dir
    }
    
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def test_model_strategy_combination(model_instance: QuantizedHuggingFaceLLM, model_name: str, 
                                  strategy_name: str, contexts: Dict, patch_generator, logger) -> Dict:
    """
    Test a single model-strategy combination with timeout
    
    Args:
        model_instance: The quantized model instance
        model_name: Name of the model
        strategy_name: Name of the strategy
        contexts: Dictionary of contexts
        patch_generator: Patch generator instance
        logger: Logger instance
        
    Returns:
        Result dictionary
    """
    try:
        # Check if context exists for this strategy
        if strategy_name not in contexts:
            raise ValueError(f"No context available for strategy: {strategy_name}")
        
        # Get model info for logging
        model_info = model_instance.get_model_info()
        logger.info(f"Using {model_name} with {model_info['quantization_strategy']} quantization")
        
        # Generate patch using the quantized model's generate_patch method
        with timeout_context(600, f"{model_name}-{strategy_name} patch generation"):  # 10 minute timeout
            # Use the quantized model's built-in generate_patch method
            patch, metrics = model_instance.generate_patch(
                context=contexts[strategy_name],
                strategy=strategy_name
            )
        
        result = {
            'patch': patch,
            'metrics': metrics,
            'model_info': model_info,
            'context_length': len(contexts[strategy_name]) if contexts[strategy_name] else 0,
            'success': metrics.get('success', True) if metrics else False,
            'timestamp': datetime.now().isoformat(),
            'quantization_strategy': model_info.get('quantization_strategy', 'unknown')
        }
        
        return result
        
    except TimeoutError as e:
        error_msg = str(e)
        logger.error(f"Timeout with {model_name}-{strategy_name}: {error_msg}")
        return {
            'error': error_msg,
            'success': False,
            'timeout': True,
            'timestamp': datetime.now().isoformat(),
            'quantization_strategy': getattr(model_instance, 'quantization_strategy', 'unknown') if model_instance else 'unknown'
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error with {model_name}-{strategy_name}: {error_msg}")
        return {
            'error': error_msg,
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'quantization_strategy': getattr(model_instance, 'quantization_strategy', 'unknown') if model_instance else 'unknown'
        }


def create_model_with_fallback(model_name: str, model_config: Dict[str, Any], logger) -> Optional[QuantizedHuggingFaceLLM]:
    """
    Create a quantized model with fallback strategies
    
    Args:
        model_name: Name of the model
        model_config: Model configuration
        logger: Logger instance
        
    Returns:
        QuantizedHuggingFaceLLM instance or None if all attempts fail
    """
    try:
        # Try to create the quantized model
        model_instance = create_quantized_model(model_name, model_config)
        
        # Log model information
        model_info = model_instance.get_model_info()
        logger.info(f"✓ Created {model_name} with {model_info['quantization_strategy']} quantization")
        logger.info(f"  - Load time: {model_info['load_time']:.2f}s")
        if 'model_size_gb' in model_info:
            logger.info(f"  - Model size: {model_info['model_size_gb']:.2f}GB")
        
        return model_instance
        
    except Exception as e:
        logger.error(f"Failed to create quantized model {model_name}: {str(e)}")
        
        # Try with more aggressive quantization as fallback
        try:
            logger.info(f"Attempting fallback with forced 4-bit quantization for {model_name}")
            fallback_config = model_config.copy()
            fallback_config.update({
                'force_quantization': True,
                'use_4bit': True,
                'max_memory_gb': 4  # Very conservative memory limit
            })
            
            model_instance = create_quantized_model(model_name, fallback_config)
            model_info = model_instance.get_model_info()
            logger.info(f"✓ Created {model_name} with fallback 4-bit quantization")
            
            return model_instance
            
        except Exception as fallback_error:
            logger.error(f"Fallback also failed for {model_name}: {str(fallback_error)}")
            return None


def main():
    """Main execution function with comprehensive error handling"""
    print("=" * 80)
    print("SWE Benchmark Multi-Strategy Patch Generator (Quantized)")
    print("=" * 80)
    
    overall_start_time = time.time()
    
    # Validate dependencies
    if not validate_dependencies():
        print("✗ Dependency validation failed. Exiting.")
        sys.exit(1)
    
    # Setup
    try:
        output_dir = create_output_dir()
        logger = setup_logging(output_dir)
        print(f"✓ Output directory: {output_dir}")
    except Exception as e:
        print(f"✗ Setup failed: {str(e)}")
        sys.exit(1)
    
    # Initialize components
    try:
        print("Initializing components...")
        context_builder = ContextBuilder()
        patch_generator = PatchGenerator()
        analyzer = ResultsAnalyzer()
        print("✓ Components initialized")
    except Exception as e:
        print(f"✗ Component initialization failed: {str(e)}")
        logger.error(f"Component initialization failed: {str(e)}")
        sys.exit(1)
    
    # Convert Entity objects to dictionaries
    print("\nConverting entities to proper format...")
    try:
        entities_dicts = convert_entities_to_dicts(updated_entities, problem_stmt, weights)
        if not entities_dicts:
            print("⚠ No entities were processed successfully - continuing with empty entities")
            entities_dicts = []
    except Exception as e:
        print(f"✗ Entity conversion failed: {str(e)}")
        logger.error(f"Entity conversion failed: {str(e)}")
        entities_dicts = []
    
    # Load structured data
    print("\nLoading structured data...")
    try:
        structured_data = load_structured_data()
    except Exception as e:
        print(f"⚠ Structured data loading failed: {str(e)}")
        structured_data = {}
    
    # Build contexts for all strategies
    print("\nBuilding contexts for all strategies...")
    try:
        with timeout_context(600, "context building"):  # 10 minute timeout
            contexts = context_builder.build_all_contexts(
                problem_statement=problem_stmt,
                entities=entities_dicts,
                call_hierarchy=all_hierarchy,
                structured_data=structured_data
            )
        print(f"✓ Built contexts for {len(contexts)} strategies")
    except TimeoutError as e:
        print(f"✗ Context building timed out: {str(e)}")
        logger.error(f"Context building timed out: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Context building failed: {str(e)}")
        logger.error(f"Context building failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # Test all model-strategy combinations
    results = {}
    total_combinations = len(MODEL_CONFIGS) * len(STRATEGIES)
    completed_combinations = 0
    combination_start_time = time.time()
    
    print(f"\nTesting {total_combinations} model-strategy combinations with quantization...")
    
    for model_idx, (model_name, model_config) in enumerate(MODEL_CONFIGS.items(), 1):
        print(f"\n{'='*50}")
        print(f"Testing Model {model_idx}/{len(MODEL_CONFIGS)}: {model_name}")
        print(f"{'='*50}")
        
        model_results = {}
        model_instance = None
        
        try:
            # Create quantized model instance with timeout
            try:
                with timeout_context(300, f"model {model_name} initialization"):  # 5 minute timeout
                    model_instance = create_model_with_fallback(model_name, model_config, logger)
                
                if model_instance is None:
                    raise Exception("Failed to create model with all fallback strategies")
                    
                print(f"✓ Created quantized model: {model_name}")
                
            except TimeoutError as e:
                print(f"✗ Model initialization timed out: {str(e)}")
                # Create timeout entries for all strategies
                for strategy_name in STRATEGIES:
                    model_results[strategy_name] = {
                        'error': f"Model initialization timeout: {str(e)}",
                        'success': False,
                        'timeout': True,
                        'timestamp': datetime.now().isoformat()
                    }
                    completed_combinations += 1
                continue
            except Exception as e:
                print(f"✗ Failed to initialize model {model_name}: {str(e)}")
                # Create failure entries for all strategies
                for strategy_name in STRATEGIES:
                    model_results[strategy_name] = {
                        'error': f"Model initialization failed: {str(e)}",
                        'success': False,
                        'timestamp': datetime.now().isoformat()
                    }
                    completed_combinations += 1
                continue
            
            # Test each strategy
            for strategy_idx, strategy_name in enumerate(STRATEGIES, 1):
                progress_tracker(completed_combinations, total_combinations, 
                               "Testing combinations", combination_start_time)
                
                print(f"\nStrategy {strategy_idx}/{len(STRATEGIES)}: {strategy_name}")
                print("-" * 30)
                
                strategy_start_time = time.time()
                
                result = test_model_strategy_combination(
                    model_instance, model_name, strategy_name, 
                    contexts, patch_generator, logger
                )
                
                model_results[strategy_name] = result
                
                # Print result with quantization info
                if result.get('success', False):
                    patch_length = len(result.get('patch', ''))
                    elapsed = time.time() - strategy_start_time
                    quant_strategy = result.get('quantization_strategy', 'unknown')
                    print(f"✓ Generated patch ({patch_length} chars) in {elapsed:.1f}s [{quant_strategy}]")
                elif result.get('timeout', False):
                    print(f"⏰ Timed out: {result.get('error', 'Unknown timeout')}")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    print(f"✗ Failed: {error_msg}")
                
                completed_combinations += 1
                
                # Clean up GPU memory after each generation
                cleanup_gpu_memory()
            
        except Exception as e:
            error_msg = f"Model processing failed: {str(e)}"
            logger.error(f"Failed to process model {model_name}: {error_msg}")
            
            # Create failure entries for remaining strategies
            for strategy_name in STRATEGIES:
                if strategy_name not in model_results:
                    model_results[strategy_name] = {
                        'error': error_msg,
                        'success': False,
                        'timestamp': datetime.now().isoformat()
                    }
                    completed_combinations += 1
            
            print(f"✗ Model processing failed: {str(e)}")
        
        finally:
            # Clean up model instance
            if model_instance:
                del model_instance
                cleanup_gpu_memory()
        
        results[model_name] = model_results
    
    progress_tracker(completed_combinations, total_combinations, 
                   "Testing combinations", combination_start_time)
    print(f"\n✓ Completed all {completed_combinations} combinations")
    
    # Analyze results
    print(f"\n{'='*50}")
    print("Analyzing Results")
    print(f"{'='*50}")
    
    try:
        with timeout_context(300, "results analysis"):  # 5 minute timeout
            analysis = analyzer.analyze_results(results)
        print("✓ Results analysis completed")
    except TimeoutError as e:
        print(f"⚠ Results analysis timed out: {str(e)}")
        logger.error(f"Results analysis timed out: {str(e)}")
        analysis = {'error': str(e), 'timestamp': datetime.now().isoformat()}
    except Exception as e:
        print(f"⚠ Results analysis failed: {str(e)}")
        logger.error(f"Results analysis failed: {str(e)}")
        analysis = {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    # Save everything
    print(f"\n{'='*50}")
    print("Saving Results")
    print(f"{'='*50}")
    
    save_results(results, analysis, contexts, output_dir)
    
    # Final cleanup
    cleanup_gpu_memory()
    
    # Print final summary with quantization info
    total_elapsed = time.time() - overall_start_time
    successful_patches = sum(
        1 for model_results in results.values()
        for result in model_results.values()
        if result.get('success', False)
    )
    
    # Count quantization strategies used
    quant_strategies = {}
    for model_results in results.values():
        for result in model_results.values():
            strategy = result.get('quantization_strategy', 'unknown')
            quant_strategies[strategy] = quant_strategies.get(strategy, 0) + 1
    
    print(f"\n{'='*50}")
    print("EXECUTION COMPLETE!")
    print(f"{'='*50}")
    print(f"Results saved to: {output_dir}")
    print(f"Total execution time: {total_elapsed/60:.1f} minutes")
    print(f"Combinations tested: {completed_combinations}/{total_combinations}")
    print(f"Successful patches: {successful_patches}")
    print(f"Success rate: {(successful_patches/completed_combinations)*100:.1f}%" if completed_combinations > 0 else "N/A")
    print(f"Quantization strategies used:")
    for strategy, count in quant_strategies.items():
        print(f"  - {strategy}: {count} combinations")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        cleanup_gpu_memory()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        traceback.print_exc()
        cleanup_gpu_memory()
        sys.exit(1)