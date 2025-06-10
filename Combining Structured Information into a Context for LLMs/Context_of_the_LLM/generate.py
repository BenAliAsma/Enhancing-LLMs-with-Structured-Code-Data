import argparse
import sys
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.patch_generator import SWEPatchGenerator
from src.visualization import SWEVisualization
from src.utils import setup_logging, create_directory_structure
from src.config import MODEL_CONFIGS, STRATEGIES

def get_model_parameters(model_name):
    """Get model parameter count from HuggingFace or estimate"""
    parameter_estimates = {
        # DeepSeek models
        'deepseek-ai/deepseek-coder-1.3b-instruct': 1_300_000_000,
        'deepseek-ai/deepseek-coder-6.7b-instruct': 6_700_000_000,
        
        # CodeLlama models
        'codellama/CodeLlama-7b-Instruct-hf': 7_000_000_000,
        'codellama/CodeLlama-13b-Instruct-hf': 13_000_000_000,
        
        # WizardCoder models
        'WizardLM/WizardCoder-Python-7B-V1.0': 7_000_000_000,
        'WizardLM/WizardCoder-15B-V1.0': 15_000_000_000,
        
        # CodeGen models
        'Salesforce/codegen-350M-mono': 350_000_000,
        'Salesforce/codegen-2B-mono': 2_000_000_000,
        'Salesforce/codegen-6B-mono': 6_000_000_000,
        
        # StarCoder models
        'bigcode/starcoder-1b': 1_000_000_000,
        'bigcode/starcoder-15.5b': 15_500_000_000,
        'bigcode/tiny_starcoder_py': 164_000_000,
        
        # Other models
        'microsoft/DialoGPT-small': 117_000_000,
        'microsoft/codebert-base': 125_000_000,
        'Salesforce/codet5p-770m': 770_000_000,
        'ise-uiuc/Magicoder-S-DS-6.7B': 6_700_000_000,
    }
    
    return parameter_estimates.get(model_name, 0)

def format_number(num):
    """Format large numbers with appropriate suffixes"""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(num)

def create_detailed_statistics_table(generator, results, args):
    """Create comprehensive statistics table with model information"""
    
    # Collect model statistics
    model_stats = []
    
    for model_key in args.models:
        model_config = MODEL_CONFIGS.get(model_key, {})
        model_name = model_config.get('name', model_key)
        
        # Get model-specific results
        model_results = {k: v for k, v in results.items() if k.startswith(model_key)}
        
        if not model_results:
            continue
            
        # Calculate aggregated metrics for this model
        total_patches = len(model_results)
        successful_patches = sum(1 for r in model_results.values() if r.get('success', False))
        total_time = sum(r.get('generation_time', 0) for r in model_results.values())
        total_tokens_generated = sum(r.get('patch_length', 0) for r in model_results.values())
        total_chars_generated = sum(len(r.get('patch', '')) for r in model_results.values())
        
        avg_time_per_patch = total_time / max(total_patches, 1)
        avg_tokens_per_patch = total_tokens_generated / max(total_patches, 1)
        avg_chars_per_patch = total_chars_generated / max(total_patches, 1)
        success_rate = (successful_patches / max(total_patches, 1)) * 100
        
        # Calculate tokens per second
        tokens_per_second = total_tokens_generated / max(total_time, 0.001)
        chars_per_second = total_chars_generated / max(total_time, 0.001)
        
        # Get quality scores
        quality_scores = [r.get('quality_score', 0) for r in model_results.values()]
        avg_quality = sum(quality_scores) / max(len(quality_scores), 1)
        
        # Get model parameters
        parameters = get_model_parameters(model_name)
        
        # Get max tokens from config
        max_tokens_config = model_config.get('max_tokens', 2048)
        
        # Find actual max tokens generated
        max_tokens_generated = max((len(r.get('patch', '').split()) for r in model_results.values()), default=0)
        max_chars_generated = max((len(r.get('patch', '')) for r in model_results.values()), default=0)
        
        # Determine model status
        has_errors = any('error' in r for r in model_results.values())
        status = "✅ Working" if successful_patches > 0 else "❌ Failed" if has_errors else "⚠️ Low Quality"
        
        model_stats.append({
            'Model': model_key,
            'Full Name': model_name,
            'Parameters': format_number(parameters),
            'Parameters (Raw)': parameters,
            'Status': status,
            'Max Tokens (Config)': max_tokens_config,
            'Max Tokens Generated': max_tokens_generated,
            'Max Chars Generated': max_chars_generated,
            'Total Patches': total_patches,
            'Successful Patches': successful_patches,
            'Success Rate (%)': f"{success_rate:.1f}%",
            'Total Time (s)': f"{total_time:.2f}",
            'Avg Time/Patch (s)': f"{avg_time_per_patch:.2f}",
            'Total Tokens Generated': total_tokens_generated,
            'Avg Tokens/Patch': f"{avg_tokens_per_patch:.1f}",
            'Total Chars Generated': total_chars_generated,
            'Avg Chars/Patch': f"{avg_chars_per_patch:.1f}",
            'Tokens/Second': f"{tokens_per_second:.2f}",
            'Chars/Second': f"{chars_per_second:.1f}",
            'Avg Quality Score': f"{avg_quality:.3f}",
            'Model Type': model_config.get('type', 'unknown'),
            'Temperature': model_config.get('temperature', 0.7),
        })
    
    # Create DataFrame
    df = pd.DataFrame(model_stats)
    
    # Sort by success rate and parameters
    if not df.empty:
        df = df.sort_values(['Success Rate (%)', 'Parameters (Raw)'], ascending=[False, False])
    
    return df

def print_formatted_table(df):
    """Print a nicely formatted table"""
    if df.empty:
        print("❌ No model statistics available")
        return
        
    print("\n" + "="*120)
    print("📊 DETAILED MODEL PERFORMANCE STATISTICS")
    print("="*120)
    
    # Print basic info table
    basic_cols = ['Model', 'Parameters', 'Status', 'Successful Patches', 'Success Rate (%)', 'Avg Time/Patch (s)']
    if all(col in df.columns for col in basic_cols):
        print("\n🏆 QUICK OVERVIEW:")
        print("-" * 80)
        print(df[basic_cols].to_string(index=False))
    
    # Print performance table
    perf_cols = ['Model', 'Tokens/Second', 'Chars/Second', 'Avg Quality Score', 'Max Tokens Generated']
    if all(col in df.columns for col in perf_cols):
        print("\n⚡ PERFORMANCE METRICS:")
        print("-" * 80)
        print(df[perf_cols].to_string(index=False))
    
    # Print detailed configuration
    config_cols = ['Model', 'Full Name', 'Model Type', 'Max Tokens (Config)', 'Temperature']
    if all(col in df.columns for col in config_cols):
        print("\n⚙️  MODEL CONFIGURATION:")
        print("-" * 80)
        print(df[config_cols].to_string(index=False))
    
    # Print generation statistics
    gen_cols = ['Model', 'Total Patches', 'Total Time (s)', 'Total Tokens Generated', 'Total Chars Generated']
    if all(col in df.columns for col in gen_cols):
        print("\n📈 GENERATION STATISTICS:")
        print("-" * 80)
        print(df[gen_cols].to_string(index=False))

def save_statistics_files(df, results_dir, generator):
    """Save statistics in multiple formats"""
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    csv_file = results_path / f"model_statistics_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"💾 Statistics saved as CSV: {csv_file}")
    
    # Save as JSON
    json_file = results_path / f"model_statistics_{timestamp}.json"
    df.to_json(json_file, orient='records', indent=2)
    print(f"💾 Statistics saved as JSON: {json_file}")
    
    # Save as Excel (if openpyxl is available)
    try:
        excel_file = results_path / f"model_statistics_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Model Statistics', index=False)
            
            # Add performance summary sheet
            summary_data = generator.get_performance_summary()
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_excel(writer, sheet_name='Overall Summary', index=False)
            
        print(f"💾 Statistics saved as Excel: {excel_file}")
    except ImportError:
        print("⚠️  Excel export requires openpyxl: pip install openpyxl")
    
    # Save detailed markdown report
    md_file = results_path / f"model_statistics_report_{timestamp}.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# SWE Benchmark Model Performance Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Quick Overview\n\n")
        basic_cols = ['Model', 'Parameters', 'Status', 'Successful Patches', 'Success Rate (%)', 'Avg Time/Patch (s)']
        if all(col in df.columns for col in basic_cols):
            f.write(df[basic_cols].to_markdown(index=False))
        
        f.write("\n\n## Performance Metrics\n\n")
        perf_cols = ['Model', 'Tokens/Second', 'Chars/Second', 'Avg Quality Score', 'Max Tokens Generated']
        if all(col in df.columns for col in perf_cols):
            f.write(df[perf_cols].to_markdown(index=False))
        
        f.write("\n\n## Full Statistics\n\n")
        f.write(df.to_markdown(index=False))
        
        # Add recommendations
        f.write("\n\n## Recommendations\n\n")
        if not df.empty:
            best_model = df.iloc[0]['Model']
            f.write(f"- **Best Overall Model:** {best_model}\n")
            
            fastest_model = df.loc[df['Tokens/Second'].str.replace('[^\d.]', '', regex=True).astype(float).idxmax(), 'Model']
            f.write(f"- **Fastest Model:** {fastest_model}\n")
            
            if 'Avg Quality Score' in df.columns:
                highest_quality = df.loc[df['Avg Quality Score'].str.replace('[^\d.]', '', regex=True).astype(float).idxmax(), 'Model']
                f.write(f"- **Highest Quality:** {highest_quality}\n")
    
    print(f"💾 Detailed report saved as Markdown: {md_file}")
    
    return {
        'csv': csv_file,
        'json': json_file,
        'markdown': md_file,
        'excel': excel_file if 'excel_file' in locals() else None
    }

def main():
    parser = argparse.ArgumentParser(description="SWE Benchmark Patch Generator")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--results-dir", default="results", help="Results directory path")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS.keys()), 
                       default=list(MODEL_CONFIGS.keys()), help="Models to use")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, 
                       default=STRATEGIES, help="Strategies to use")
    parser.add_argument("--skip-generation", action="store_true", 
                       help="Skip patch generation, only visualize existing results")
    parser.add_argument("--visualize", action="store_true", 
                       help="Generate visualizations")
    parser.add_argument("--report", action="store_true", 
                       help="Generate analysis report")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level=getattr(logging, args.log_level))
    
    # Create directory structure
    create_directory_structure(".")
    
    print("🚀 SWE Benchmark Patch Generator")
    print("=" * 50)
    
    results = {}
    generator = None
    
    if not args.skip_generation:
        # Initialize generator
        generator = SWEPatchGenerator(args.data_dir)
        
        # Filter models and strategies
        if args.models != list(MODEL_CONFIGS.keys()):
            generator.models = {k: v for k, v in MODEL_CONFIGS.items() if k in args.models}
        
        print(f"📊 Configuration:")
        print(f"   Models: {args.models}")
        print(f"   Strategies: {args.strategies}")
        print(f"   Data Dir: {args.data_dir}")
        print(f"   Results Dir: {args.results_dir}")
        print()
        
        # Initialize models
        generator.initialize_models()
        
        # Generate patches
        print("🔧 Generating patches...")
        results = generator.generate_all_patches()
        
        # Save results
        print("💾 Saving results...")
        generator.save_results(args.results_dir)
        
        # Print summary
        summary = generator.get_performance_summary()
        print(f"\n📈 Generation Summary:")
        print(f"   Total generations: {summary['total_generations']}")
        print(f"   Successful: {summary['successful_generations']}")
        print(f"   Average time: {summary['average_generation_time']:.2f}s")
        print(f"   Average tokens/sec: {summary['average_tokens_per_second']:.2f}")
    
    else:
        # Load existing results for analysis
        try:
            results_path = Path(args.results_dir)
            summary_file = results_path / "generation_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                    results = data.get('results', {})
                    print(f"📂 Loaded existing results: {len(results)} entries")
        except Exception as e:
            print(f"⚠️ Could not load existing results: {e}")
    
    # Generate detailed statistics table
    if results:
        print("\n🔍 Generating detailed model statistics...")
        stats_df = create_detailed_statistics_table(generator or SWEPatchGenerator(args.data_dir), results, args)
        
        # Print formatted table
        print_formatted_table(stats_df)
        
        # Save statistics files
        saved_files = save_statistics_files(stats_df, args.results_dir, generator or SWEPatchGenerator(args.data_dir))
        
        print(f"\n📁 Statistics files saved:")
        for file_type, file_path in saved_files.items():
            if file_path:
                print(f"   {file_type.upper()}: {file_path}")
    
    # Visualization
    if args.visualize:
        print("\n📊 Generating visualizations...")
        viz = SWEVisualization(args.results_dir)
        
        # Create plots
        viz.plot_performance_comparison(
            os.path.join(args.results_dir, "performance_comparison.png")
        )
        viz.plot_success_rate_heatmap(
            os.path.join(args.results_dir, "success_rate_heatmap.png")
        )
        
        print("✓ Visualizations saved")
    
    # Report generation
    if args.report:
        print("\n📋 Generating analysis report...")
        viz = SWEVisualization(args.results_dir)
        report = viz.generate_report(
            os.path.join(args.results_dir, "analysis_report.md")
        )
        print("✓ Report saved")
        print(f"\nReport preview:\n{report[:500]}...")
    
    print("\n✅ Process completed successfully!")
    print(f"📁 Check {args.results_dir} for all outputs")

if __name__ == "__main__":
    import logging
    main()