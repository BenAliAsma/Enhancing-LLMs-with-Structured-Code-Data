#!/usr/bin/env python3
"""
Results Analyzer for SWE Benchmark Patch Generator
Analyzes and compares results across different model-strategy combinations
Now includes git patch validation using git apply
Enhanced with successful patch counts per model
"""

import json
import statistics
import subprocess
import tempfile
import os
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from pathlib import Path

# Import test_patch safely
try:
    from src.config import test_patch
except ImportError:
    test_patch = None


class PatchValidator:
    """Validates patches using git apply"""
    
    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = repo_path or os.getcwd()
        self.temp_dirs = []
    
    def validate_patch(self, patch_content: str, test_repo_path: Optional[str] = None) -> Tuple[bool, str]:
        """
        Validate a patch using git apply
        
        Args:
            patch_content: The patch content as a string
            test_repo_path: Optional path to test repository
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not patch_content or not patch_content.strip():
            return False, "Empty patch content"
        
        # Create a temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        try:
            # Use provided repo path or current directory
            source_repo = test_repo_path or self.repo_path
            
            # Copy the repository to temp directory
            if os.path.exists(source_repo) and os.path.isdir(source_repo):
                # Copy git repository
                shutil.copytree(source_repo, os.path.join(temp_dir, 'test_repo'))
                test_repo_dir = os.path.join(temp_dir, 'test_repo')
            else:
                # Initialize a new git repo for testing
                test_repo_dir = os.path.join(temp_dir, 'test_repo')
                os.makedirs(test_repo_dir)
                subprocess.run(['git', 'init'], cwd=test_repo_dir, capture_output=True)
                subprocess.run(['git', 'config', 'user.email', 'test@example.com'], cwd=test_repo_dir, capture_output=True)
                subprocess.run(['git', 'config', 'user.name', 'Test User'], cwd=test_repo_dir, capture_output=True)
            
            # Write patch to temporary file
            patch_file = os.path.join(temp_dir, 'test.patch')
            with open(patch_file, 'w', encoding='utf-8') as f:
                f.write(patch_content)
            
            # Try to apply the patch with git apply --check first
            result = subprocess.run(
                ['git', 'apply', '--check', patch_file],
                cwd=test_repo_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Patch syntax is valid, now try to actually apply it
                apply_result = subprocess.run(
                    ['git', 'apply', patch_file],
                    cwd=test_repo_dir,
                    capture_output=True,
                    text=True
                )
                
                if apply_result.returncode == 0:
                    return True, "Patch applied successfully"
                else:
                    return False, f"Patch application failed: {apply_result.stderr}"
            else:
                return False, f"Invalid patch format: {result.stderr}"
                
        except subprocess.CalledProcessError as e:
            return False, f"Git command failed: {e.stderr}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
                if temp_dir in self.temp_dirs:
                    self.temp_dirs.remove(temp_dir)
            except Exception:
                pass  # Ignore cleanup errors
    
    def cleanup(self):
        """Clean up any remaining temporary directories"""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception:
                pass
        self.temp_dirs.clear()


class ResultsAnalyzer:
    """Analyzes patch generation results across models and strategies"""
    
    def __init__(self, repo_path: Optional[str] = None):
        self.results = {}
        self.analysis = {}
        self.patch_validator = PatchValidator(repo_path)
    
    def analyze_results(self, results: Dict[str, Dict[str, Any]], validate_patches: bool = True) -> Dict[str, Any]:
        """
        Comprehensive analysis of all results with optional patch validation
        
        Args:
            results: Dictionary of {model_name: {strategy_name: result_data}}
            validate_patches: Whether to validate patches using git apply
            
        Returns:
            Dictionary containing comprehensive analysis
        """
        self.results = results
        
        # Validate patches if requested
        if validate_patches:
            self._validate_all_patches()
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'validation_enabled': validate_patches,
            'summary': self._generate_summary(),
            'success_rates': self._calculate_success_rates(),
            'patch_validation_results': self._analyze_patch_validation() if validate_patches else {},
            'performance_metrics': self._analyze_performance_metrics(),
            'strategy_comparison': self._compare_strategies(),
            'model_comparison': self._compare_models(),
            'model_patch_counts': self._analyze_model_patch_counts(),  # New method
            'context_analysis': self._analyze_context_usage(),
            'patch_analysis': self._analyze_patches(),
            'recommendations': self._generate_recommendations(),
            'failure_analysis': self._analyze_failures()
        }
        
        self.analysis = analysis
        return analysis
    
    def _validate_all_patches(self):
        """Validate all patches in the results using git apply"""
        print("Validating patches using git apply...")
        
        for model_name, model_results in self.results.items():
            for strategy, result in model_results.items():
                if 'patch' in result and result['patch']:
                    print(f"Validating patch for {model_name} - {strategy}")
                    
                    is_valid, error_msg = self.patch_validator.validate_patch(result['patch'])
                    
                    # Update result with validation info
                    result['patch_valid'] = is_valid
                    result['patch_validation_error'] = error_msg if not is_valid else None
                    
                    # Update success status based on patch validity
                    if not is_valid:
                        result['success'] = False
                        if 'error' not in result:
                            result['error'] = f"Invalid patch: {error_msg}"
                    
                    print(f"  Result: {'VALID' if is_valid else 'INVALID'}")
                    if not is_valid:
                        print(f"  Error: {error_msg}")
    
    def _analyze_patch_validation(self) -> Dict[str, Any]:
        """Analyze patch validation results"""
        total_patches = 0
        valid_patches = 0
        validation_errors = Counter()
        
        validation_by_model = defaultdict(lambda: {'valid': 0, 'invalid': 0, 'total': 0})
        validation_by_strategy = defaultdict(lambda: {'valid': 0, 'invalid': 0, 'total': 0})
        
        for model_name, model_results in self.results.items():
            for strategy, result in model_results.items():
                if 'patch' in result and result['patch']:
                    total_patches += 1
                    validation_by_model[model_name]['total'] += 1
                    validation_by_strategy[strategy]['total'] += 1
                    
                    if result.get('patch_valid', False):
                        valid_patches += 1
                        validation_by_model[model_name]['valid'] += 1
                        validation_by_strategy[strategy]['valid'] += 1
                    else:
                        validation_by_model[model_name]['invalid'] += 1
                        validation_by_strategy[strategy]['invalid'] += 1
                        
                        # Categorize validation error
                        error_msg = result.get('patch_validation_error', 'Unknown error')
                        error_category = self._categorize_patch_error(error_msg)
                        validation_errors[error_category] += 1
        
        # Calculate validation rates
        for model_data in validation_by_model.values():
            model_data['validation_rate'] = model_data['valid'] / model_data['total'] if model_data['total'] > 0 else 0
        
        for strategy_data in validation_by_strategy.values():
            strategy_data['validation_rate'] = strategy_data['valid'] / strategy_data['total'] if strategy_data['total'] > 0 else 0
        
        return {
            'total_patches_tested': total_patches,
            'valid_patches': valid_patches,
            'invalid_patches': total_patches - valid_patches,
            'overall_validation_rate': valid_patches / total_patches if total_patches > 0 else 0,
            'validation_errors': dict(validation_errors),
            'by_model': dict(validation_by_model),
            'by_strategy': dict(validation_by_strategy),
            'common_validation_errors': validation_errors.most_common(5)
        }
    
    def _categorize_patch_error(self, error_message: str) -> str:
        """Categorize patch validation error messages"""
        error_lower = error_message.lower()
        
        if 'does not exist' in error_lower or 'no such file' in error_lower:
            return 'file_not_found'
        elif 'does not apply' in error_lower or 'patch does not apply' in error_lower:
            return 'patch_does_not_apply'
        elif 'malformed patch' in error_lower or 'corrupt patch' in error_lower:
            return 'malformed_patch'
        elif 'already applied' in error_lower or 'already exists' in error_lower:
            return 'already_applied'
        elif 'whitespace' in error_lower:
            return 'whitespace_error'
        elif 'hunk' in error_lower and 'failed' in error_lower:
            return 'hunk_failed'
        elif 'empty patch' in error_lower:
            return 'empty_patch'
        else:
            return 'other_patch_error'
    
    def _analyze_model_patch_counts(self) -> Dict[str, Any]:
        """Analyze successful patch counts for each model"""
        model_patch_stats = {}
        
        for model_name, model_results in self.results.items():
            successful_patches = 0
            valid_patches = 0
            total_patches = 0
            failed_patches = 0
            strategies_with_patches = []
            
            for strategy, result in model_results.items():
                if 'patch' in result and result['patch']:
                    total_patches += 1
                    
                    if result.get('success', False):
                        successful_patches += 1
                        strategies_with_patches.append(strategy)
                        
                        if result.get('patch_valid', False):
                            valid_patches += 1
                    else:
                        failed_patches += 1
            
            model_patch_stats[model_name] = {
                'successful_patches': successful_patches,
                'valid_patches': valid_patches,
                'total_patches': total_patches,
                'failed_patches': failed_patches,
                'success_rate': successful_patches / total_patches if total_patches > 0 else 0,
                'validation_rate': valid_patches / successful_patches if successful_patches > 0 else 0,
                'strategies_with_successful_patches': strategies_with_patches,
                'successful_patch_count_by_strategy': len(strategies_with_patches)
            }
        
        # Sort models by successful patch count
        sorted_by_success = sorted(
            model_patch_stats.items(), 
            key=lambda x: x[1]['successful_patches'], 
            reverse=True
        )
        
        # Sort models by valid patch count
        sorted_by_valid = sorted(
            model_patch_stats.items(), 
            key=lambda x: x[1]['valid_patches'], 
            reverse=True
        )
        
        return {
            'by_model': model_patch_stats,
            'ranked_by_successful_patches': sorted_by_success,
            'ranked_by_valid_patches': sorted_by_valid,
            'total_successful_patches': sum(stats['successful_patches'] for stats in model_patch_stats.values()),
            'total_valid_patches': sum(stats['valid_patches'] for stats in model_patch_stats.values())
        }
    
    def test_with_config_patch(self):
        """Test the validator with the patch from config.py"""
        if test_patch is None:
            print("No test_patch available from config.py")
            return False, "No test patch available"
        
        print("Testing with patch from config.py...")
        is_valid, error_msg = self.patch_validator.validate_patch(test_patch)
        
        print(f"Config patch validation result: {'VALID' if is_valid else 'INVALID'}")
        if not is_valid:
            print(f"Error: {error_msg}")
        
        return is_valid, error_msg
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate high-level summary statistics"""
        total_combinations = 0
        successful_combinations = 0
        total_models = len(self.results)
        total_strategies = 0
        patches_generated = 0
        valid_patches = 0
        
        # Get total strategies from the first model's results, or default to 0
        if self.results:
            first_model_results = next(iter(self.results.values()))
            total_strategies = len(first_model_results)
        
        for model_name, model_results in self.results.items():
            for strategy, result in model_results.items():
                total_combinations += 1
                if result.get('success', False):
                    successful_combinations += 1
                if 'patch' in result and result['patch']:
                    patches_generated += 1
                    if result.get('patch_valid', False):
                        valid_patches += 1
        
        return {
            'total_models': total_models,
            'total_strategies': total_strategies,
            'total_combinations': total_combinations,
            'successful_combinations': successful_combinations,
            'patches_generated': patches_generated,
            'valid_patches': valid_patches,
            'overall_success_rate': successful_combinations / total_combinations if total_combinations > 0 else 0,
            'patch_validation_rate': valid_patches / patches_generated if patches_generated > 0 else 0,
            'models_tested': list(self.results.keys())
        }
    
    def _calculate_success_rates(self) -> Dict[str, Any]:
        """Calculate success rates by model and strategy"""
        model_success = {}
        strategy_success = defaultdict(lambda: {'successful': 0, 'total': 0})
        
        # Calculate per-model success rates
        for model_name, model_results in self.results.items():
            successful = sum(1 for result in model_results.values() if result.get('success', False))
            total = len(model_results)
            model_success[model_name] = {
                'successful': successful,
                'total': total,
                'rate': successful / total if total > 0 else 0
            }
        
        # Calculate per-strategy success rates
        for model_results in self.results.values():
            for strategy, result in model_results.items():
                strategy_success[strategy]['total'] += 1
                if result.get('success', False):
                    strategy_success[strategy]['successful'] += 1
        
        # Add success rates to strategy data
        for strategy in strategy_success:
            data = strategy_success[strategy]
            data['rate'] = data['successful'] / data['total'] if data['total'] > 0 else 0
        
        return {
            'by_model': model_success,
            'by_strategy': dict(strategy_success)
        }
    
    def _analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics like generation time, token usage, etc."""
        metrics_by_model = {}
        metrics_by_strategy = defaultdict(list)
        
        for model_name, model_results in self.results.items():
            model_metrics = []
            
            for strategy, result in model_results.items():
                if result.get('success') and 'metrics' in result and result['metrics']:
                    metrics = result['metrics']
                    model_metrics.append(metrics)
                    metrics_by_strategy[strategy].append(metrics)
            
            if model_metrics:
                metrics_by_model[model_name] = self._aggregate_metrics(model_metrics)
        
        # Aggregate metrics by strategy
        strategy_metrics = {}
        for strategy, metrics_list in metrics_by_strategy.items():
            if metrics_list:
                strategy_metrics[strategy] = self._aggregate_metrics(metrics_list)
        
        return {
            'by_model': metrics_by_model,
            'by_strategy': strategy_metrics
        }
    
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict[str, Any]:
        """Aggregate a list of metrics dictionaries"""
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # Common metrics to aggregate
        numeric_fields = ['generation_time', 'tokens_used', 'cost', 'response_length']
        
        for field in numeric_fields:
            values = [m.get(field) for m in metrics_list if m.get(field) is not None]
            if values:
                aggregated[field] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0
                }
        
        return aggregated
    
    def _compare_strategies(self) -> Dict[str, Any]:
        """Compare strategies across all models"""
        strategy_stats = defaultdict(lambda: {
            'successes': 0,
            'failures': 0,
            'patch_lengths': [],
            'context_lengths': [],
            'valid_patches': 0,
            'models_tested': set()
        })
        
        for model_name, model_results in self.results.items():
            for strategy, result in model_results.items():
                stats = strategy_stats[strategy]
                stats['models_tested'].add(model_name)
                
                if result.get('success'):
                    stats['successes'] += 1
                    if 'patch' in result and result['patch']:
                        stats['patch_lengths'].append(len(result['patch']))
                        if result.get('patch_valid'):
                            stats['valid_patches'] += 1
                else:
                    stats['failures'] += 1
                
                if 'context_length' in result:
                    stats['context_lengths'].append(result['context_length'])
        
        # Calculate derived statistics
        comparison = {}
        for strategy, stats in strategy_stats.items():
            total = stats['successes'] + stats['failures']
            patches_with_content = len(stats['patch_lengths'])
            comparison[strategy] = {
                'success_rate': stats['successes'] / total if total > 0 else 0,
                'patch_validation_rate': stats['valid_patches'] / patches_with_content if patches_with_content > 0 else 0,
                'total_attempts': total,
                'models_tested': len(stats['models_tested']),
                'avg_patch_length': statistics.mean(stats['patch_lengths']) if stats['patch_lengths'] else 0,
                'avg_context_length': statistics.mean(stats['context_lengths']) if stats['context_lengths'] else 0
            }
        
        # Rank strategies by success rate
        ranked = sorted(comparison.items(), key=lambda x: x[1]['success_rate'], reverse=True)
        
        return {
            'detailed': comparison,
            'ranked_by_success': ranked
        }
    
    def _compare_models(self) -> Dict[str, Any]:
        """Compare models across all strategies"""
        model_stats = {}
        
        for model_name, model_results in self.results.items():
            successes = sum(1 for r in model_results.values() if r.get('success'))
            total = len(model_results)
            
            patch_lengths = []
            context_lengths = []
            valid_patches = 0
            patches_with_content = 0
            
            for result in model_results.values():
                if result.get('success') and 'patch' in result and result['patch']:
                    patch_lengths.append(len(result['patch']))
                    patches_with_content += 1
                    if result.get('patch_valid'):
                        valid_patches += 1
                if 'context_length' in result:
                    context_lengths.append(result['context_length'])
            
            model_stats[model_name] = {
                'success_rate': successes / total if total > 0 else 0,
                'patch_validation_rate': valid_patches / patches_with_content if patches_with_content > 0 else 0,
                'total_attempts': total,
                'strategies_tested': len(model_results),
                'avg_patch_length': statistics.mean(patch_lengths) if patch_lengths else 0,
                'avg_context_length': statistics.mean(context_lengths) if context_lengths else 0
            }
        
        # Rank models by success rate
        ranked = sorted(model_stats.items(), key=lambda x: x[1]['success_rate'], reverse=True)
        
        return {
            'detailed': model_stats,
            'ranked_by_success': ranked
        }
    
    def _analyze_context_usage(self) -> Dict[str, Any]:
        """Analyze context length usage patterns"""
        context_data = []
        context_by_strategy = defaultdict(list)
        context_by_model = defaultdict(list)
        
        for model_name, model_results in self.results.items():
            for strategy, result in model_results.items():
                if 'context_length' in result:
                    length = result['context_length']
                    success = result.get('success', False)
                    
                    context_data.append({
                        'length': length,
                        'success': success,
                        'model': model_name,
                        'strategy': strategy
                    })
                    
                    context_by_strategy[strategy].append(length)
                    context_by_model[model_name].append(length)
        
        if not context_data:
            return {}
        
        lengths = [d['length'] for d in context_data]
        successful_lengths = [d['length'] for d in context_data if d['success']]
        failed_lengths = [d['length'] for d in context_data if not d['success']]
        
        analysis = {
            'overall_stats': {
                'mean': statistics.mean(lengths),
                'median': statistics.median(lengths),
                'min': min(lengths),
                'max': max(lengths)
            },
            'success_correlation': {
                'successful_mean': statistics.mean(successful_lengths) if successful_lengths else 0,
                'failed_mean': statistics.mean(failed_lengths) if failed_lengths else 0
            },
            'by_strategy': {k: statistics.mean(v) for k, v in context_by_strategy.items()},
            'by_model': {k: statistics.mean(v) for k, v in context_by_model.items()}
        }
        
        return analysis
    
    def _analyze_patches(self) -> Dict[str, Any]:
        """Analyze generated patches"""
        patch_data = []
        
        for model_name, model_results in self.results.items():
            for strategy, result in model_results.items():
                if 'patch' in result and result['patch']:
                    patch = result['patch']
                    patch_data.append({
                        'length': len(patch),
                        'model': model_name,
                        'strategy': strategy,
                        'lines': patch.count('\n') + 1,
                        'has_imports': 'import ' in patch,
                        'has_functions': 'def ' in patch,
                        'has_classes': 'class ' in patch,
                        'is_valid': result.get('patch_valid', False)
                    })
        
        if not patch_data:
            return {}
        
        lengths = [p['length'] for p in patch_data]
        lines = [p['lines'] for p in patch_data]
        valid_patches = [p for p in patch_data if p['is_valid']]
        
        return {
            'total_patches': len(patch_data),
            'valid_patches': len(valid_patches),
            'length_stats': {
                'mean': statistics.mean(lengths),
                'median': statistics.median(lengths),
                'min': min(lengths),
                'max': max(lengths)
            },
            'line_stats': {
                'mean': statistics.mean(lines),
                'median': statistics.median(lines),
                'min': min(lines),
                'max': max(lines)
            },
            'content_analysis': {
                'patches_with_imports': sum(1 for p in patch_data if p['has_imports']),
                'patches_with_functions': sum(1 for p in patch_data if p['has_functions']),
                'patches_with_classes': sum(1 for p in patch_data if p['has_classes'])
            },
            'valid_patch_stats': {
                'avg_length': statistics.mean([p['length'] for p in valid_patches]) if valid_patches else 0,
                'avg_lines': statistics.mean([p['lines'] for p in valid_patches]) if valid_patches else 0
            }
        }
    
    def _analyze_failures(self) -> Dict[str, Any]:
        """Analyze failure patterns"""
        failures = []
        error_types = Counter()
        failures_by_model = defaultdict(list)
        failures_by_strategy = defaultdict(list)
        
        for model_name, model_results in self.results.items():
            for strategy, result in model_results.items():
                if not result.get('success') and 'error' in result:
                    error = result['error']
                    failures.append({
                        'model': model_name,
                        'strategy': strategy,
                        'error': error
                    })
                    
                    # Categorize error type
                    error_type = self._categorize_error(error)
                    error_types[error_type] += 1
                    
                    failures_by_model[model_name].append(error_type)
                    failures_by_strategy[strategy].append(error_type)
        
        return {
            'total_failures': len(failures),
            'error_types': dict(error_types),
            'failures_by_model': {k: dict(Counter(v)) for k, v in failures_by_model.items()},
            'failures_by_strategy': {k: dict(Counter(v)) for k, v in failures_by_strategy.items()},
            'common_errors': error_types.most_common(5)
        }
    
    def _categorize_error(self, error_message: str) -> str:
        """Categorize error message into common types"""
        error_lower = error_message.lower()
        
        if 'invalid patch' in error_lower:
            return 'invalid_patch'
        elif 'timeout' in error_lower:
            return 'timeout'
        elif 'rate limit' in error_lower or 'quota' in error_lower:
            return 'rate_limit'
        elif 'token' in error_lower and ('limit' in error_lower or 'max' in error_lower):
            return 'token_limit'
        elif 'connection' in error_lower or 'network' in error_lower:
            return 'network'
        elif 'auth' in error_lower or 'key' in error_lower:
            return 'authentication'
        elif 'parse' in error_lower or 'json' in error_lower:
            return 'parsing'
        else:
            return 'other'
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if not self.analysis:
            return recommendations
        
        # Success rate recommendations
        success_rates = self.analysis.get('success_rates', {})
        if 'by_model' in success_rates:
            model_rates = [(model, data['rate']) for model, data in success_rates['by_model'].items()]
            if model_rates:
                best_model = max(model_rates, key=lambda x: x[1])
                recommendations.append(f"Model '{best_model[0]}' shows highest success rate ({best_model[1]:.2%})")
        
        if 'by_strategy' in success_rates:
            strategy_rates = [(strategy, data['rate']) for strategy, data in success_rates['by_strategy'].items()]
            if strategy_rates:
                best_strategy = max(strategy_rates, key=lambda x: x[1])
                recommendations.append(f"Strategy '{best_strategy[0]}' shows highest success rate ({best_strategy[1]:.2%})")
        
        # Patch count recommendations
        model_patch_counts = self.analysis.get('model_patch_counts', {})
        if 'ranked_by_successful_patches' in model_patch_counts and model_patch_counts['ranked_by_successful_patches']:
            best_patch_model = model_patch_counts['ranked_by_successful_patches'][0]
            recommendations.append(f"Model '{best_patch_model[0]}' generated the most successful patches ({best_patch_model[1]['successful_patches']} patches)")
        
        # Patch validation recommendations
        patch_validation = self.analysis.get('patch_validation_results', {})
        if patch_validation and 'overall_validation_rate' in patch_validation:
            validation_rate = patch_validation['overall_validation_rate']
            if validation_rate < 0.5:
                recommendations.append(f"Low patch validation rate ({validation_rate:.2%}) - consider improving patch generation")
            elif validation_rate > 0.8:
                recommendations.append(f"High patch validation rate ({validation_rate:.2%}) - good patch quality")
        
        # Context usage recommendations
        context_analysis = self.analysis.get('context_analysis', {})
        if 'success_correlation' in context_analysis:
            successful_mean = context_analysis['success_correlation']['successful_mean']
            failed_mean = context_analysis['success_correlation']['failed_mean']
            if successful_mean > failed_mean * 1.1:
                recommendations.append("Longer contexts tend to produce more successful patches")
            elif failed_mean > successful_mean * 1.1:
                recommendations.append("Shorter contexts may be more effective")
        
        # Failure analysis recommendations
        failure_analysis = self.analysis.get('failure_analysis', {})
        if 'common_errors' in failure_analysis and failure_analysis['common_errors']:
            top_error = failure_analysis['common_errors'][0]
            recommendations.append(f"Address '{top_error[0]}' errors - they account for {top_error[1]} failures")
        
        return recommendations
    
    def print_summary(self):
        """Print a formatted summary of the analysis"""
        if not self.analysis:
            print("No analysis available. Run analyze_results() first.")
            return
        
        summary = self.analysis['summary']
        
        print("\n" + "="*60)
        print("RESULTS ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Models Tested: {summary['total_models']}")
        print(f"Strategies Tested: {summary['total_strategies']}")
        print(f"Total Combinations: {summary['total_combinations']}")
        print(f"Successful: {summary['successful_combinations']}")
        print(f"Overall Success Rate: {summary['overall_success_rate']:.2%}")
        
        # Patch validation results
        patch_validation = self.analysis.get('patch_validation_results', {})
        if patch_validation:
            print(f"Patches Generated: {summary['patches_generated']}")
            print(f"Valid Patches: {summary['valid_patches']}")
            print(f"Patch Validation Rate: {summary['patch_validation_rate']:.2%}")
        
        # Top performing model and strategy
        model_comparison = self.analysis.get('model_comparison', {})
        if 'ranked_by_success' in model_comparison and model_comparison['ranked_by_success']:
            best_model = model_comparison['ranked_by_success'][0]
            print(f"Best Model: {best_model[0]} ({best_model[1]['success_rate']:.2%})")
        
        strategy_comparison = self.analysis.get('strategy_comparison', {})
        if 'ranked_by_success' in strategy_comparison and strategy_comparison['ranked_by_success']:
            best_strategy = strategy_comparison['ranked_by_success'][0]
            print(f"Best Strategy: {best_strategy[0]} ({best_strategy[1]['success_rate']:.2%})")
        
        # Recommendations
        recommendations = self.analysis.get('recommendations', [])
        if recommendations:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        
        print("="*60)


if __name__ == "__main__":
    # Example usage
    analyzer = ResultsAnalyzer()
    
    # Mock results for testing
    mock_results = {
        "gpt-4": {
            "focused": {"success": True, "patch": "code here", "context_length": 1000},
            "comprehensive": {"success": False, "error": "Token limit exceeded"}
        },
        "claude-3": {
            "focused": {"success": True, "patch": "different code", "context_length": 800},
            "comprehensive": {"success": True, "patch": "more code", "context_length": 1200}
        }
    }
    
    analysis = analyzer.analyze_results(mock_results)
    analyzer.print_summary()