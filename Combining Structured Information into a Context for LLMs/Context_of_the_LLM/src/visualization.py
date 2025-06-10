import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import json
import os

class SWEVisualization:
    """Visualization utilities for SWE benchmark results"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.metrics = []
        self.load_metrics()
        
    def load_metrics(self):
        """Load metrics from results directory"""
        metrics_path = os.path.join(self.results_dir, "all_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
    
    def plot_performance_comparison(self, save_path: str = None):
        """Plot performance comparison across models and strategies"""
        if not self.metrics:
            print("No metrics available for plotting")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.metrics)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SWE Benchmark Performance Analysis', fontsize=16)
        
        # 1. Generation time by model
        if 'model' in df.columns and 'generation_time' in df.columns:
            sns.boxplot(data=df, x='model', y='generation_time', ax=axes[0,0])
            axes[0,0].set_title('Generation Time by Model')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Tokens per second by model
        if 'model' in df.columns and 'tokens_per_second' in df.columns:
            sns.boxplot(data=df, x='model', y='tokens_per_second', ax=axes[0,1])
            axes[0,1].set_title('Tokens per Second by Model')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Context length by strategy
        if 'strategy' in df.columns and 'context_length' in df.columns:
            sns.boxplot(data=df, x='strategy', y='context_length', ax=axes[1,0])
            axes[1,0].set_title('Context Length by Strategy')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Output tokens by strategy
        if 'strategy' in df.columns and 'output_tokens' in df.columns:
            sns.boxplot(data=df, x='strategy', y='output_tokens', ax=axes[1,1])
            axes[1,1].set_title('Output Tokens by Strategy')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_success_rate_heatmap(self, save_path: str = None):
        """Plot success rate heatmap"""
        if not self.metrics:
            print("No metrics available for plotting")
            return
        
        df = pd.DataFrame(self.metrics)
        
        # Create pivot table
        if 'model' in df.columns and 'strategy' in df.columns and 'success' in df.columns:
            pivot_table = df.pivot_table(
                values='success', 
                index='model', 
                columns='strategy', 
                aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', fmt='.2f', 
                       cbar_kws={'label': 'Success Rate'})
            plt.title('Model-Strategy Success Rate Heatmap')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_report(self, save_path: str = None):
        """Generate comprehensive analysis report"""
        if not self.metrics:
            return "No metrics available for report generation"
        
        df = pd.DataFrame(self.metrics)
        
        report = f"""# SWE Benchmark Patch Generator Report
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- Total Generations: {len(df)}
- Successful Generations: {df['success'].sum() if 'success' in df.columns else 'N/A'}
- Average Generation Time: {df['generation_time'].mean():.2f}s (if available)
- Average Tokens/Second: {df['tokens_per_second'].mean():.2f} (if available)

## Model Performance
"""
        
        if 'model' in df.columns:
            model_stats = df.groupby('model').agg({
                'generation_time': ['mean', 'std'],
                'tokens_per_second': ['mean', 'std'],
                'success': 'mean'
            }).round(3)
            
            report += f"\n{model_stats.to_string()}\n"
        
        report += "\n## Strategy Analysis\n"
        
        if 'strategy' in df.columns:
            strategy_stats = df.groupby('strategy').agg({
                'context_length': ['mean', 'std'],
                'output_tokens': ['mean', 'std']
            }).round(3)
            
            report += f"\n{strategy_stats.to_string()}\n"
        
        report += "\n## Recommendations\n"
        
        if 'model' in df.columns and 'generation_time' in df.columns:
            fastest_model = df.groupby('model')['generation_time'].mean().idxmin()
            report += f"- Fastest Model: {fastest_model}\n"
        
        if 'strategy' in df.columns and 'context_length' in df.columns:
            most_efficient_strategy = df.groupby('strategy')['context_length'].mean().idxmin()
            report += f"- Most Efficient Strategy: {most_efficient_strategy}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report