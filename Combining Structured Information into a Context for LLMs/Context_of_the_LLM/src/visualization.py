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
                try:
                    self.metrics = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error reading {metrics_path}: Invalid JSON format")
                    self.metrics = []
    
    def plot_performance_comparison(self, save_path: str = None):
        """Plot performance comparison across models and strategies"""
        if not self.metrics:
            print("No metrics available for plotting")
            return
        
        df = pd.DataFrame(self.metrics)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SWE Benchmark Performance Analysis', fontsize=16)

        # 1. Generation time by model
        if {'model', 'generation_time'}.issubset(df.columns):
            sns.boxplot(data=df, x='model', y='generation_time', ax=axes[0,0])
            axes[0,0].set_title('Generation Time by Model')
            axes[0,0].tick_params(axis='x', rotation=45)

        # 2. Tokens per second by model
        if {'model', 'tokens_per_second'}.issubset(df.columns):
            sns.boxplot(data=df, x='model', y='tokens_per_second', ax=axes[0,1])
            axes[0,1].set_title('Tokens per Second by Model')
            axes[0,1].tick_params(axis='x', rotation=45)

        # 3. Context length by strategy
        if {'strategy', 'context_length'}.issubset(df.columns):
            sns.boxplot(data=df, x='strategy', y='context_length', ax=axes[1,0])
            axes[1,0].set_title('Context Length by Strategy')
            axes[1,0].tick_params(axis='x', rotation=45)

        # 4. Output tokens by strategy
        if {'strategy', 'output_tokens'}.issubset(df.columns):
            sns.boxplot(data=df, x='strategy', y='output_tokens', ax=axes[1,1])
            axes[1,1].set_title('Output Tokens by Strategy')
            axes[1,1].tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_success_rate_heatmap(self, save_path: str = None):
        """Plot success rate heatmap"""
        if not self.metrics:
            print("No metrics available for plotting")
            return
        
        df = pd.DataFrame(self.metrics)

        if {'model', 'strategy', 'success'}.issubset(df.columns):
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
        else:
            print("Missing columns for heatmap: 'model', 'strategy', or 'success'")
    
    def generate_report(self, save_path: str = None) -> str:
        """Generate comprehensive analysis report"""
        if not self.metrics:
            return "No metrics available for report generation"
        
        df = pd.DataFrame(self.metrics)
        lines = [
            "# SWE Benchmark Patch Generator Report",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Summary Statistics",
            f"- Total Generations: {len(df)}",
            f"- Successful Generations: {df['success'].sum() if 'success' in df.columns else 'N/A'}",
            f"- Average Generation Time: {df['generation_time'].mean():.2f}s" if 'generation_time' in df.columns else "- Average Generation Time: N/A",
            f"- Average Tokens/Second: {df['tokens_per_second'].mean():.2f}" if 'tokens_per_second' in df.columns else "- Average Tokens/Second: N/A",
            "\n## Model Performance"
        ]

        if 'model' in df.columns:
            try:
                model_stats = df.groupby('model').agg({
                    'generation_time': ['mean', 'std'] if 'generation_time' in df.columns else [],
                    'tokens_per_second': ['mean', 'std'] if 'tokens_per_second' in df.columns else [],
                    'success': 'mean' if 'success' in df.columns else []
                }).round(3)
                lines.append(f"\n{model_stats.to_string()}")
            except Exception as e:
                lines.append(f"\nError computing model performance: {e}")

        lines.append("\n## Strategy Analysis")

        if 'strategy' in df.columns:
            try:
                strategy_stats = df.groupby('strategy').agg({
                    'context_length': ['mean', 'std'] if 'context_length' in df.columns else [],
                    'output_tokens': ['mean', 'std'] if 'output_tokens' in df.columns else []
                }).round(3)
                lines.append(f"\n{strategy_stats.to_string()}")
            except Exception as e:
                lines.append(f"\nError computing strategy analysis: {e}")

        lines.append("\n## Recommendations")

        try:
            if 'model' in df.columns and 'generation_time' in df.columns:
                fastest_model = df.groupby('model')['generation_time'].mean().idxmin()
                lines.append(f"- Fastest Model: {fastest_model}")
        except Exception:
            lines.append("- Fastest Model: Unable to determine")

        try:
            if 'strategy' in df.columns and 'context_length' in df.columns:
                most_efficient_strategy = df.groupby('strategy')['context_length'].mean().idxmin()
                lines.append(f"- Most Efficient Strategy: {most_efficient_strategy}")
        except Exception:
            lines.append("- Most Efficient Strategy: Unable to determine")

        report = "\n".join(lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)

        return report
