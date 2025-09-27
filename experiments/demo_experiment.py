"""
Main experiment script demonstrating how to use the DAG scheduling framework.

This script shows various ways to run experiments and analyze results.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from src.dag_generators import DAGFactory
from src.schedulers import SchedulerFactory  
from src.cost_matrices import generate_cost_matrices
from src.experiment_runner import ExperimentRunner, ExperimentConfig
from src.visualization import DAGVisualizer, ScheduleVisualizer, ResultsVisualizer
from config.experiment_configs import DAG_CONFIGS, PLATFORM_CONFIGS, CCR_VALUES


def run_single_example():
    """Run a single scheduling example to demonstrate the framework."""
    print("=" * 60)
    print("RUNNING SINGLE EXAMPLE")
    print("=" * 60)
    
    # Generate a Gaussian Elimination DAG
    dag_generator = DAGFactory.create_generator('gaussian')
    graph, task_list, message_list = dag_generator.generate(chi=4)
    
    print(f"Generated DAG with {len(task_list)} tasks and {len(message_list)} messages")
    
    # Generate cost matrices
    ET, CT, _, _ = generate_cost_matrices(
        graph, num_processors=2, num_buses=2, ccr=1.0, random_state=42
    )
    
    # Run CC-TMS
    cctms_scheduler = SchedulerFactory.create_scheduler('cctms')
    cctms_result = cctms_scheduler.schedule(graph, task_list, message_list, ET, CT)
    
    print(f"CC-TMS Makespan: {cctms_result['makespan']:.2f} ms")
    
    # Run QL-CC-TMS
    qlcctms_scheduler = SchedulerFactory.create_scheduler('qlcctms')
    qlcctms_result = qlcctms_scheduler.schedule(
        graph, task_list, message_list, ET, CT, random_state=42
    )
    
    print(f"QL-CC-TMS Makespan: {qlcctms_result['makespan']:.2f} ms")
    print(f"Q-Learning Episodes: {qlcctms_result.get('q_learning_episodes', 'N/A')}")
    print(f"Q-Learning Converged: {qlcctms_result.get('q_learning_converged', 'N/A')}")
    
    # Calculate makespan ratio
    ratio = (cctms_result['makespan'] / qlcctms_result['makespan']) * 100
    print(f"Makespan Ratio (CC-TMS/QL-CC-TMS × 100): {ratio:.2f}%")
    
    # Visualize DAG (comment out if running without display)
    # DAGVisualizer.visualize_dag(graph, "Gaussian Elimination DAG (χ=4)")
    
    # Visualize schedules (comment out if running without display)
    # ScheduleVisualizer.plot_gantt_chart(cctms_result, task_list, message_list, "CC-TMS Schedule")
    # ScheduleVisualizer.plot_gantt_chart(qlcctms_result, task_list, message_list, "QL-CC-TMS Schedule")


def run_small_experiment():
    """Run a small-scale experiment for testing."""
    print("\n" + "=" * 60)
    print("RUNNING SMALL-SCALE EXPERIMENT")
    print("=" * 60)
    
    # Create experiment configuration
    config = ExperimentConfig(
        dag_type='gaussian',
        dag_params={'chi': [3, 4]},
        processors=[2, 4],
        buses=[1, 2],
        ccr_values=[0.5, 1.0],
        algorithms=['cctms', 'qlcctms'],
        num_runs=5,  # Small number for testing
        random_state_base=42
    )
    
    # Run experiment
    runner = ExperimentRunner(results_dir="../results")
    
    def progress_callback(current, total):
        if current % 10 == 0 or current == total:
            print(f"Progress: {current}/{total} ({100*current/total:.1f}%)")
    
    print("Starting experiment...")
    results = runner.run_experiment(config, progress_callback)
    
    # Save results
    results_file = runner.save_results(results, "small_test_experiment")
    print(f"Results saved to: {results_file}")
    
    # Calculate makespan ratios
    ratios_df = runner.calculate_makespan_ratios(results)
    if not ratios_df.empty:
        avg_ratio = ratios_df['makespan_ratio'].mean()
        print(f"Average Makespan Ratio: {avg_ratio:.2f}%")
    
    # Generate summary
    summary = runner.generate_summary_statistics(results)
    print(f"Total experiments run: {summary['total_experiments']}")
    
    for alg, stats in summary['by_algorithm'].items():
        print(f"{alg.upper()}: Avg Makespan = {stats['avg_makespan']:.2f} ms, "
              f"Avg Execution Time = {stats['avg_execution_time']:.4f} s")


def validate_dag_generators():
    """Validate DAG generators against theoretical counts."""
    print("\n" + "=" * 60)
    print("VALIDATING DAG GENERATORS")
    print("=" * 60)
    
    # Test each DAG type
    test_cases = [
        ('gaussian', 'chi', [3, 4, 5]),
        ('epigenomics', 'gamma', [2, 3, 4]),
        ('laplace', 'phi', [2, 3, 4]),
        ('stencil', 'xi', [2, 3, 4])
    ]
    
    for dag_type, param_name, param_values in test_cases:
        print(f"\nTesting {dag_type.upper()} DAG:")
        generator = DAGFactory.create_generator(dag_type)
        
        for param_val in param_values:
            # Generate DAG
            if dag_type == 'gaussian':
                graph, tasks, messages = generator.generate(chi=param_val)
                expected_tasks, expected_messages = generator.get_theoretical_counts(chi=param_val)
            elif dag_type == 'epigenomics':
                graph, tasks, messages = generator.generate(gamma=param_val)
                expected_tasks, expected_messages = generator.get_theoretical_counts(gamma=param_val)
            elif dag_type == 'laplace':
                graph, tasks, messages = generator.generate(phi=param_val)
                expected_tasks, expected_messages = generator.get_theoretical_counts(phi=param_val)
            elif dag_type == 'stencil':
                graph, tasks, messages = generator.generate(xi=param_val)
                expected_tasks, expected_messages = generator.get_theoretical_counts(xi=param_val)
            
            actual_tasks = len(tasks)
            actual_messages = len(messages)
            
            print(f"  {param_name}={param_val}: Tasks {actual_tasks}/{expected_tasks}, "
                  f"Messages {actual_messages}/{expected_messages} "
                  f"{'✓' if actual_tasks == expected_tasks and actual_messages == expected_messages else '✗'}")


def demonstrate_visualizations():
    """Demonstrate visualization capabilities."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION OF VISUALIZATIONS")
    print("=" * 60)
    
    # Note: Visualizations require display and may not work in all environments
    print("Visualization functions are available but commented out in this demo.")
    print("Uncomment the visualization calls in the functions above to see plots.")
    print("\nAvailable visualizations:")
    print("- DAGVisualizer.visualize_dag() - Shows DAG structure")
    print("- ScheduleVisualizer.plot_gantt_chart() - Shows scheduling results")
    print("- ResultsVisualizer.plot_makespan_comparison() - Compares algorithm performance")
    print("- ResultsVisualizer.plot_makespan_ratios() - Shows performance ratios")
    print("- ResultsVisualizer.plot_heatmap() - Shows parameter sensitivity")
    print("- ResultsVisualizer.plot_execution_time_comparison() - Compares execution times")


if __name__ == "__main__":
    print("DAG Scheduling Framework Demonstration")
    print("=" * 60)
    
    # Run demonstrations
    run_single_example()
    validate_dag_generators()
    run_small_experiment()
    demonstrate_visualizations()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nTo run larger experiments, modify the experiment configurations")
    print("in config/experiment_configs.py and use the ExperimentRunner class.")
    print("\nFor visualization, uncomment the plotting calls and ensure you have")
    print("matplotlib, seaborn, and other visualization dependencies installed.")