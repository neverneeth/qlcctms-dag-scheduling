"""
Comprehensive experiment script for running all benchmark experiments
described in the research paper.

This script runs the full experimental evaluation with all four DAG types,
various platform configurations, and CCR values.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import time
from datetime import datetime
from src.experiment_runner import ExperimentRunner, ExperimentConfig
from config.experiment_configs import (
    DAG_CONFIGS, PLATFORM_CONFIGS, CCR_VALUES, 
    ALGORITHM_CONFIGS, TIME_CONFIGS, RUN_CONFIGS
)


def run_comprehensive_experiments():
    """Run comprehensive experiments for all DAG types."""
    print("=" * 80)
    print("COMPREHENSIVE DAG SCHEDULING EXPERIMENTS")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    runner = ExperimentRunner(results_dir="./results")
    all_results = []
    
    # Progress tracking
    total_experiments = len(DAG_CONFIGS)
    current_exp = 0
    
    for dag_name, dag_config in DAG_CONFIGS.items():
        current_exp += 1
        print(f"\n[{current_exp}/{total_experiments}] Running {dag_name} experiments...")
        print(f"Description: {dag_config['description']}")
        
        # Create experiment configuration
        exp_config = ExperimentConfig(
            dag_type=dag_config['dag_type'],
            dag_params=dag_config['dag_params'],
            processors=PLATFORM_CONFIGS['processors'],
            buses=PLATFORM_CONFIGS['buses'],
            ccr_values=CCR_VALUES,
            algorithms=ALGORITHM_CONFIGS['algorithms'],
            num_runs=RUN_CONFIGS['num_runs_full'],
            et_min=TIME_CONFIGS['et_min'],
            et_max=TIME_CONFIGS['et_max'],
            ct_min=TIME_CONFIGS['ct_min'],
            ct_max=TIME_CONFIGS['ct_max'],
            random_state_base=RUN_CONFIGS['random_state_base']
        )
        
        # Calculate total runs for this experiment
        param_key = list(dag_config['dag_params'].keys())[0]
        param_values = dag_config['dag_params'][param_key]
        total_runs_this_exp = (
            len(param_values) * 
            len(PLATFORM_CONFIGS['processors']) * 
            len(PLATFORM_CONFIGS['buses']) * 
            len(CCR_VALUES) * 
            len(ALGORITHM_CONFIGS['algorithms']) * 
            RUN_CONFIGS['num_runs_full']
        )
        
        print(f"Total runs for this experiment: {total_runs_this_exp:,}")
        
        def progress_callback(current, total):
            if current % 100 == 0 or current == total:
                percent = 100 * current / total
                print(f"  Progress: {current:,}/{total:,} ({percent:.1f}%)")
        
        # Run experiment
        start_time = time.time()
        results = runner.run_experiment(exp_config, progress_callback)
        end_time = time.time()
        
        print(f"  Completed in {end_time - start_time:.1f} seconds")
        print(f"  Generated {len(results)} results")
        
        # Save individual experiment results
        results_file = runner.save_results(results, f"full_{dag_name}_experiment")
        print(f"  Results saved to: {os.path.basename(results_file)}")
        
        # Calculate and display summary statistics
        summary = runner.generate_summary_statistics(results)
        print(f"  Algorithm performance summary:")
        for alg, stats in summary['by_algorithm'].items():
            print(f"    {alg.upper()}: Avg Makespan = {stats['avg_makespan']:.2f} ms")
        
        # Calculate makespan ratios
        ratios_df = runner.calculate_makespan_ratios(results)
        if not ratios_df.empty:
            avg_ratio = ratios_df['makespan_ratio'].mean()
            std_ratio = ratios_df['makespan_ratio'].std()
            print(f"  Average Makespan Ratio (CC-TMS/QL-CC-TMS): {avg_ratio:.2f}% ± {std_ratio:.2f}%")
        
        all_results.extend(results)
    
    # Save combined results
    print(f"\n" + "=" * 80)
    print("SAVING COMBINED RESULTS")
    print("=" * 80)
    
    combined_file = runner.save_results(all_results, "comprehensive_all_dags_experiment")
    print(f"Combined results saved to: {os.path.basename(combined_file)}")
    
    # Generate overall summary
    overall_summary = runner.generate_summary_statistics(all_results)
    overall_ratios = runner.calculate_makespan_ratios(all_results)
    
    print(f"\nOverall Experiment Summary:")
    print(f"Total experiment runs: {len(all_results):,}")
    print(f"DAG types tested: {len(overall_summary['by_dag_type'])}")
    print(f"Algorithms compared: {len(overall_summary['by_algorithm'])}")
    
    if not overall_ratios.empty:
        avg_ratio = overall_ratios['makespan_ratio'].mean()
        std_ratio = overall_ratios['makespan_ratio'].std()
        min_ratio = overall_ratios['makespan_ratio'].min()
        max_ratio = overall_ratios['makespan_ratio'].max()
        
        print(f"\nOverall Makespan Ratio Statistics:")
        print(f"  Mean: {avg_ratio:.2f}%")
        print(f"  Std Dev: {std_ratio:.2f}%")
        print(f"  Min: {min_ratio:.2f}%")
        print(f"  Max: {max_ratio:.2f}%")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return combined_file, overall_summary


def run_quick_validation():
    """Run a quick validation experiment with reduced parameters."""
    print("=" * 80)
    print("QUICK VALIDATION EXPERIMENT")
    print("=" * 80)

    runner = ExperimentRunner(results_dir="./results")

    # Run a quick test with minimal parameters
    quick_config = ExperimentConfig(
        dag_type='gaussian',
        dag_params={'chi': [3, 4]},
        processors=[2, 4],
        buses=[1, 2],
        ccr_values=[1.0],
        algorithms=['cctms', 'qlcctms'],
        num_runs=10,
        random_state_base=42
    )
    
    print("Running quick validation...")
    results = runner.run_experiment(quick_config)
    
    # Save and analyze results
    results_file = runner.save_results(results, "quick_validation")
    summary = runner.generate_summary_statistics(results)
    ratios_df = runner.calculate_makespan_ratios(results)
    
    print(f"Quick validation completed!")
    print(f"Results: {len(results)} experiment runs")
    print(f"Results saved to: {os.path.basename(results_file)}")
    
    if not ratios_df.empty:
        avg_ratio = ratios_df['makespan_ratio'].mean()
        print(f"Average Makespan Ratio: {avg_ratio:.2f}%")
    
    return results_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run DAG scheduling experiments')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='Experiment mode: quick validation or full experiments')
    parser.add_argument('--results-dir', default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        print("Running quick validation experiment...")
        results_file = run_quick_validation()
        print(f"\nQuick validation complete. Results in: {results_file}")
        
    elif args.mode == 'full':
        print("Running comprehensive experiments...")
        print("WARNING: This will take a very long time (potentially hours/days)")
        print("Consider running in batches or on a high-performance computing cluster")
        
        response = input("Continue with full experiments? (y/N): ")
        if response.lower() == 'y':
            combined_file, summary = run_comprehensive_experiments()
            print(f"\nComprehensive experiments complete. Results in: {combined_file}")
        else:
            print("Full experiments cancelled.")
    
    print("\nExperiment complete!")