"""
Results Analysis Script

Analyzes experiment results and generates plots and statistics.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from src.visualization import ResultsVisualizer
from src.experiment_runner import ExperimentRunner


def analyze_results(results_file):
    """Analyze experiment results from a CSV file."""
    print("=" * 80)
    print("RESULTS ANALYSIS")
    print("=" * 80)
    
    # Load results
    print(f"Loading results from: {results_file}")
    df = pd.read_csv(results_file)
    
    print(f"Loaded {len(df)} experiment results")
    print(f"Columns: {list(df.columns)}")
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"DAG types: {df['dag_type'].unique()}")
    print(f"Algorithms: {df['algorithm'].unique()}")
    print(f"Processors tested: {sorted(df['num_processors'].unique())}")
    print(f"Buses tested: {sorted(df['num_buses'].unique())}")
    print(f"CCR values tested: {sorted(df['ccr'].unique())}")
    
    # Algorithm comparison
    print(f"\nAlgorithm Performance Summary:")
    for alg in df['algorithm'].unique():
        alg_data = df[df['algorithm'] == alg]
        print(f"{alg.upper()}:")
        print(f"  Mean Makespan: {alg_data['makespan'].mean():.2f} ± {alg_data['makespan'].std():.2f} ms")
        print(f"  Mean Execution Time: {alg_data['execution_time'].mean():.4f} ± {alg_data['execution_time'].std():.4f} s")
        print(f"  Makespan Range: {alg_data['makespan'].min():.2f} - {alg_data['makespan'].max():.2f} ms")
    
    # Calculate makespan ratios
    runner = ExperimentRunner()
    # Convert DataFrame back to ExperimentResult objects for ratio calculation
    from src.experiment_runner import ExperimentResult
    results_objects = []
    for _, row in df.iterrows():
        result = ExperimentResult(
            dag_type=row['dag_type'],
            dag_param_value=row['dag_param_value'],
            num_processors=row['num_processors'],
            num_buses=row['num_buses'],
            ccr=row['ccr'],
            algorithm=row['algorithm'],
            run_id=row['run_id'],
            makespan=row['makespan'],
            execution_time=row['execution_time'],
            num_tasks=row['num_tasks'],
            num_messages=row['num_messages'],
            converged=row.get('converged'),
            episodes=row.get('episodes')
        )
        results_objects.append(result)
    
    ratios_df = runner.calculate_makespan_ratios(results_objects)
    
    if not ratios_df.empty:
        print(f"\nMakespan Ratio Analysis (CC-TMS/QL-CC-TMS × 100):")
        print(f"  Mean Ratio: {ratios_df['makespan_ratio'].mean():.2f}%")
        print(f"  Std Dev: {ratios_df['makespan_ratio'].std():.2f}%")
        print(f"  Min Ratio: {ratios_df['makespan_ratio'].min():.2f}%")
        print(f"  Max Ratio: {ratios_df['makespan_ratio'].max():.2f}%")
        
        # Performance interpretation
        avg_ratio = ratios_df['makespan_ratio'].mean()
        if avg_ratio > 100:
            print(f"  → QL-CC-TMS performs {avg_ratio - 100:.1f}% better on average")
        else:
            print(f"  → CC-TMS performs {100 - avg_ratio:.1f}% better on average")
    
    # DAG type analysis
    print(f"\nPerformance by DAG Type:")
    for dag_type in df['dag_type'].unique():
        dag_data = df[df['dag_type'] == dag_type]
        print(f"{dag_type.upper()}:")
        for alg in dag_data['algorithm'].unique():
            alg_dag_data = dag_data[dag_data['algorithm'] == alg]
            print(f"  {alg}: {alg_dag_data['makespan'].mean():.2f} ± {alg_dag_data['makespan'].std():.2f} ms")
    
    # Platform sensitivity analysis
    print(f"\nPlatform Sensitivity Analysis:")
    
    # Processor impact
    print(f"Impact of Number of Processors:")
    for proc_count in sorted(df['num_processors'].unique()):
        proc_data = df[df['num_processors'] == proc_count]
        avg_makespan = proc_data['makespan'].mean()
        print(f"  {proc_count} processors: {avg_makespan:.2f} ms average makespan")
    
    # Bus impact
    print(f"Impact of Number of Buses:")
    for bus_count in sorted(df['num_buses'].unique()):
        bus_data = df[df['num_buses'] == bus_count]
        avg_makespan = bus_data['makespan'].mean()
        print(f"  {bus_count} buses: {avg_makespan:.2f} ms average makespan")
    
    # CCR impact
    print(f"Impact of CCR:")
    for ccr_val in sorted(df['ccr'].unique()):
        ccr_data = df[df['ccr'] == ccr_val]
        avg_makespan = ccr_data['makespan'].mean()
        print(f"  CCR {ccr_val}: {avg_makespan:.2f} ms average makespan")
    
    return df, ratios_df


def generate_visualizations(df, ratios_df, output_dir="../results/plots"):
    """Generate visualizations for the results."""
    print(f"\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Makespan comparison by DAG parameter
        print("Generating makespan comparison plot...")
        ResultsVisualizer.plot_makespan_comparison(
            df, 
            group_by='dag_param_value',
            save_path=os.path.join(output_dir, 'makespan_comparison.png')
        )
        
        # Makespan ratios
        if not ratios_df.empty:
            print("Generating makespan ratio plot...")
            ResultsVisualizer.plot_makespan_ratios(
                ratios_df,
                group_by='dag_param_value', 
                save_path=os.path.join(output_dir, 'makespan_ratios.png')
            )
        
        # Execution time comparison
        print("Generating execution time comparison...")
        ResultsVisualizer.plot_execution_time_comparison(
            df,
            save_path=os.path.join(output_dir, 'execution_time_comparison.png')
        )
        
        # Heatmaps for different algorithms
        for algorithm in df['algorithm'].unique():
            print(f"Generating heatmap for {algorithm}...")
            ResultsVisualizer.plot_heatmap(
                df,
                metric='makespan',
                x_col='num_processors',
                y_col='ccr',
                algorithm=algorithm,
                save_path=os.path.join(output_dir, f'heatmap_{algorithm}.png')
            )
        
        print(f"All visualizations saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        print("This might be due to missing display or matplotlib dependencies.")
        print("Visualizations are optional and analysis can continue without them.")


def export_summary_report(df, ratios_df, output_file="../results/summary_report.txt"):
    """Export a summary report to a text file."""
    print(f"Generating summary report: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("DAG SCHEDULING EXPERIMENT RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Experiments: {len(df)}\n")
        f.write(f"DAG Types: {', '.join(df['dag_type'].unique())}\n")
        f.write(f"Algorithms: {', '.join(df['algorithm'].unique())}\n")
        f.write(f"Processors Tested: {', '.join(map(str, sorted(df['num_processors'].unique())))}\n")
        f.write(f"Buses Tested: {', '.join(map(str, sorted(df['num_buses'].unique())))}\n")
        f.write(f"CCR Values: {', '.join(map(str, sorted(df['ccr'].unique())))}\n\n")
        
        f.write("ALGORITHM PERFORMANCE\n")
        f.write("-" * 25 + "\n")
        for alg in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == alg]
            f.write(f"{alg.upper()}:\n")
            f.write(f"  Mean Makespan: {alg_data['makespan'].mean():.2f} ± {alg_data['makespan'].std():.2f} ms\n")
            f.write(f"  Mean Execution Time: {alg_data['execution_time'].mean():.4f} ± {alg_data['execution_time'].std():.4f} s\n")
            f.write(f"  Makespan Range: {alg_data['makespan'].min():.2f} - {alg_data['makespan'].max():.2f} ms\n\n")
        
        if not ratios_df.empty:
            f.write("MAKESPAN RATIO ANALYSIS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Mean Ratio (CC-TMS/QL-CC-TMS × 100): {ratios_df['makespan_ratio'].mean():.2f}%\n")
            f.write(f"Standard Deviation: {ratios_df['makespan_ratio'].std():.2f}%\n")
            f.write(f"Min Ratio: {ratios_df['makespan_ratio'].min():.2f}%\n")
            f.write(f"Max Ratio: {ratios_df['makespan_ratio'].max():.2f}%\n\n")
        
        f.write("PERFORMANCE BY DAG TYPE\n")
        f.write("-" * 25 + "\n")
        for dag_type in df['dag_type'].unique():
            dag_data = df[df['dag_type'] == dag_type]
            f.write(f"{dag_type.upper()}:\n")
            for alg in dag_data['algorithm'].unique():
                alg_dag_data = dag_data[dag_data['algorithm'] == alg]
                f.write(f"  {alg}: {alg_dag_data['makespan'].mean():.2f} ± {alg_dag_data['makespan'].std():.2f} ms\n")
            f.write("\n")
    
    print(f"Summary report saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze DAG scheduling experiment results')
    parser.add_argument('results_file', help='Path to the results CSV file')
    parser.add_argument('--plots', action='store_true', help='Generate visualization plots')
    parser.add_argument('--report', action='store_true', help='Generate summary report')
    parser.add_argument('--output-dir', default='../results', help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file '{args.results_file}' not found.")
        sys.exit(1)
    
    # Analyze results
    df, ratios_df = analyze_results(args.results_file)
    
    # Generate visualizations if requested
    if args.plots:
        plots_dir = os.path.join(args.output_dir, 'plots')
        generate_visualizations(df, ratios_df, plots_dir)
    
    # Generate summary report if requested
    if args.report:
        report_file = os.path.join(args.output_dir, 'summary_report.txt')
        export_summary_report(df, ratios_df, report_file)
    
    print(f"\nAnalysis complete!")
    if args.plots:
        print(f"Plots saved to: {os.path.join(args.output_dir, 'plots')}")
    if args.report:
        print(f"Report saved to: {os.path.join(args.output_dir, 'summary_report.txt')}")