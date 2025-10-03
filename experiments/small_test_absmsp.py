""" Sanity check for ABS MAKESPAN COMPARISON EXPERIMENT """

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from collections import defaultdict
import json
from scipy import stats

# Add the parent directory to the path to import framework modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.experiment_utils import SchedulerUtils
from src.dag_generators import DAGFactory
from src.schedulers import SchedulerFactory
# Fix the import - use the actual function name from cost_matrices
from src.cost_matrices import generate_cost_matrices  # Changed this line
from src.experiment_runner import ExperimentRunner, ExperimentConfig, ExperimentResult

# Import centralized configuration
from config.constants import (
    DAGTypes, Algorithms, DEFAULT_QL_PARAMS, ExperimentConfig,
    PlotConfig, FileConfig, ProgressConfig
)


class AbsoluteMakespanComparator:
    """
    Class to handle the absolute makespan comparison experiment.
    
    This experiment compares CC-TMS and QL-CC-TMS algorithms across:
    - 4 DAG types with different parameters
    - Various platform configurations (processors × buses)
    - 100 iterations per configuration for statistical significance
    - Generates comprehensive box plots for visualization
    """
    
    def __init__(self, results_dir="./results", iterations=30):  # Reduced iterations for testing
        """
        Initialize the experiment with configuration parameters.
        
        Args:
            results_dir (str): Directory to save results
            iterations (int): Number of iterations per configuration
        """
        self.results_dir = results_dir
        self.iterations = iterations  # Use parameter value
        self.ccr = 1.0 
        
        # Experiment parameters - REDUCED for small test
        self.dag_configs = ExperimentConfig.SMALL_DAG_CONFIGS.copy()
        
        # Platform configurations - REDUCED for small test
        self.processors = ExperimentConfig.SMALL_PROCESSORS.copy()
        self.buses = ExperimentConfig.SMALL_BUSES.copy()
        
        # Algorithms to compare
        self.algorithms = [Algorithms.CCTMS, Algorithms.QLCCTMS]
        
        # Q-learning parameters
        self.ql_params = DEFAULT_QL_PARAMS.copy()
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'plots'), exist_ok=True)
        
        print(f"Small Test - Absolute Makespan Comparison Experiment Initialized")
        print(f"Results directory: {self.results_dir}")
        print(f"Iterations per configuration: {self.iterations}")
        print(f"Total configurations: {self._calculate_total_configurations()}")
        
        # Calculate expected plots
        expected_plots = len(list(self.dag_configs.values())[0]) * (len(self.processors) + len(self.buses))
        print(f"Expected plots: {expected_plots}")
        params_count = len(list(self.dag_configs.values())[0])
        print(f"  - Constant processors plots: {params_count * len(self.processors)} (QL-CC-TMS with CC-TMS reference)")
        print(f"  - Constant buses plots: {params_count * len(self.buses)} (QL-CC-TMS with CC-TMS reference)")
        print(f"  - Only QL-CC-TMS plots generated (CC-TMS shown as reference lines)")
    
    def _calculate_total_configurations(self):
        """Calculate the total number of experimental configurations."""
        total = 0
        for dag_type, params in self.dag_configs.items():
            total += len(params) * len(self.processors) * len(self.buses) * len(self.algorithms) * self.iterations
        return total
    
    def run_experiment(self):
        """
        Run the complete absolute makespan comparison experiment.
        
        This method executes all experimental configurations and collects results
        for statistical analysis and visualization.
        
        Returns:
            tuple: (results_dataframe, experiment_metadata)
        """
        print("\n" + "="*80)
        print("STARTING ABSOLUTE MAKESPAN COMPARISON EXPERIMENT")
        print("="*80)
        
        # Initialize experiment tracking
        all_results = []
        experiment_start_time = datetime.now()
        total_configs = self._calculate_total_configurations()
        current_config = 0
        
        # Create experiment runner
        runner = ExperimentRunner()
        
        # Iterate through all DAG types and their parameters
        for dag_type, param_values in self.dag_configs.items():
            print(f"\nProcessing {dag_type.upper()} DAGs...")
            
            for param_value in param_values:
                print(f"  Parameter value: {param_value}")
                
                # Generate DAG with current parameter
                dag_generator = DAGFactory.create_generator(dag_type)
                
                # Set the appropriate parameter based on DAG type
                if dag_type == 'gaussian':
                    dag, task_list, message_list = dag_generator.generate(chi=param_value)
                elif dag_type == 'epigenomics':
                    dag, task_list, message_list = dag_generator.generate(gamma=param_value)
                elif dag_type == 'laplace':
                    dag, task_list, message_list = dag_generator.generate(phi=param_value)
                elif dag_type == 'stencil':
                    dag, task_list, message_list = dag_generator.generate(lambda_val=param_value, xi=param_value)
                
                # Test across all platform configurations
                for num_proc, num_bus in product(self.processors, self.buses):
                    platform_config = f"P{num_proc}B{num_bus}"
                    print(f"    Platform: {platform_config}")
                    
                    # Generate cost matrices for current platform using the correct function
                    
                    
                    # Test both algorithms
                    for algorithm in self.algorithms:
                        
                        print(f"      Algorithm: {algorithm.upper()}")
                        
                        # Create scheduler with centralized parameters
                        # In run_experiment method:
                        # Create scheduler with centralized parameters
                        if algorithm == Algorithms.QLCCTMS:
                            scheduler = SchedulerUtils.create_scheduler_with_params(algorithm, self.ql_params)
                        else:
                            scheduler = SchedulerUtils.create_scheduler_with_params(algorithm)
                        
                        # Run multiple iterations for statistical significance
                        iteration_results = []
                        for iteration in range(self.iterations):
                            ET, CT, TL, ML = generate_cost_matrices(
                                dag,  
                                num_proc, num_bus, self.ccr,
                                random_state=42 + iteration
                            )
                            current_config += 1
                            
                            # Show progress
                            if current_config % ProgressConfig.PROGRESS_INTERVAL_SMALL == 0 or current_config == total_configs:
                                progress = (current_config / total_configs) * 100
                                print(f"        Progress: {current_config}/{total_configs} ({progress:.1f}%)")
                            
                            # Execute scheduling
                            try:
                                result = scheduler.schedule(
                                    dag, task_list, message_list, ET, CT, 
                                    random_state=42 + iteration  # Different seed per iteration
                                )
                                
                                # Store detailed result
                                experiment_result = {
                                    'dag_type': dag_type,
                                    'dag_parameter': param_value,
                                    'num_processors': num_proc,
                                    'num_buses': num_bus,
                                    'platform_config': platform_config,
                                    'algorithm': algorithm,
                                    'iteration': iteration + 1,
                                    'makespan': result['makespan'],
                                    'ccr': self.ccr,
                                    'num_tasks': len(task_list),
                                    'num_messages': len(message_list),
                                    'execution_time': result.get('execution_time', 0),  
                                    'timestamp': datetime.now().isoformat()
                                }
                                
                                # Add Q-learning specific metrics if available
                                if 'q_learning_episodes' in result:
                                    experiment_result['q_learning_episodes'] = result['q_learning_episodes']
                                    experiment_result['q_learning_converged'] = result['q_learning_converged']
                                
                                all_results.append(experiment_result)
                                iteration_results.append(result['makespan'])
                                
                            except Exception as e:
                                print(f"        Error in iteration {iteration + 1}: {e}")
                                import traceback
                                traceback.print_exc()
                                continue
                        
                        # Print summary statistics for this configuration
                        if iteration_results:
                            mean_makespan = np.mean(iteration_results)
                            std_makespan = np.std(iteration_results)
                            print(f"        {algorithm.upper()} - Mean: {mean_makespan:.2f}ms, Std: {std_makespan:.2f}ms")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Create experiment metadata
        experiment_end_time = datetime.now()
        metadata = {
            'experiment_name': 'absolute_makespan_comparison_small_test',
            'start_time': experiment_start_time.isoformat(),
            'end_time': experiment_end_time.isoformat(),
            'duration_minutes': (experiment_end_time - experiment_start_time).total_seconds() / 60,
            'total_configurations': len(all_results),
            'iterations_per_config': self.iterations,
            'dag_types': list(self.dag_configs.keys()),
            'dag_parameters': dict(self.dag_configs),
            'processors': self.processors,
            'buses': self.buses,
            'algorithms': self.algorithms,
            'ccr': self.ccr,
            'framework_version': '1.0.0'
        }
        
        print(f"\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Total results collected: {len(results_df)}")
        print(f"Total duration: {metadata['duration_minutes']:.2f} minutes")
        
        return results_df, metadata
    
    def save_results(self, results_df, metadata):
        """
        Save experiment results and metadata to files.
        
        Args:
            results_df (pd.DataFrame): Experiment results
            metadata (dict): Experiment metadata
            
        Returns:
            tuple: (csv_filepath, json_filepath)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV results
        csv_filename = f"small_test_absolute_makespan_comparison_{timestamp}.csv"
        csv_filepath = os.path.join(self.results_dir, csv_filename)
        results_df.to_csv(csv_filepath, index=False)
        
        # Save JSON metadata
        json_filename = f"small_test_absolute_makespan_comparison_{timestamp}_metadata.json"
        json_filepath = os.path.join(self.results_dir, json_filename)
        with open(json_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  CSV: {csv_filepath}")
        print(f"  Metadata: {json_filepath}")
        
        return csv_filepath, json_filepath
    
    def generate_box_plots(self, results_df):
        """
        Generate QL-CC-TMS box plots with CC-TMS reference lines.
        
        For each DAG type and parameter combination, generates:
        - QL-CC-TMS plots with constant processors, varying buses (with CC-TMS reference)
        - QL-CC-TMS plots with constant buses, varying processors (with CC-TMS reference)
        
        Args:
            results_df (pd.DataFrame): Experiment results
        """
        print(f"\n" + "="*80)
        print("GENERATING BOX PLOTS")
        print("="*80)
        print("Generating QL-CC-TMS plots with CC-TMS reference lines:")
        print("  - Constant processors, varying buses")
        print("  - Constant buses, varying processors")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        plot_count = 0
        # Update expected plots count: only QL-CC-TMS plots with CC-TMS reference
        expected_plots = len(list(self.dag_configs.values())[0]) * (len(self.processors) + len(self.buses))
        
        # Generate plots for each DAG type and parameter
        for dag_type, param_values in self.dag_configs.items():
            for param_value in param_values:
                
                # Filter data for current DAG configuration
                dag_data = results_df[
                    (results_df['dag_type'] == dag_type) & 
                    (results_df['dag_parameter'] == param_value)
                ]
                
                if dag_data.empty:
                    print(f"Warning: No data for {dag_type} with parameter {param_value}")
                    continue
                
                print(f"\nProcessing {dag_type} (parameter: {param_value})...")
                
                # 1. CONSTANT PROCESSORS, VARYING BUSES
                for proc in self.processors:
                    # Only create plots for QL-CC-TMS, but annotate with CC-TMS comparison
                    plot_count += 1
                    
                    # Filter data for constant processor and QL-CC-TMS
                    ql_proc_data = dag_data[
                        (dag_data['num_processors'] == proc) & 
                        (dag_data['algorithm'] == 'qlcctms')
                    ]
                    
                    # Filter data for constant processor and CC-TMS (for comparison)
                    cc_proc_data = dag_data[
                        (dag_data['num_processors'] == proc) & 
                        (dag_data['algorithm'] == 'cctms')
                    ]
                    
                    if ql_proc_data.empty:
                        print(f"  Warning: No QL-CC-TMS data for P{proc}")
                        continue
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Create box plot with buses on X-axis for QL-CC-TMS
                    sns.boxplot(
                        data=ql_proc_data, 
                        x='num_buses', 
                        y='makespan',
                        ax=ax,
                        color='lightcoral'
                    )
                    
                    # Customize plot
                    title = f"{dag_type.replace('_', ' ').title()} (Param: {param_value}) - {proc} Processors - QL-CC-TMS vs CC-TMS"
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    ax.set_xlabel('Number of Buses', fontsize=12)
                    ax.set_ylabel('Makespan (ms)', fontsize=12)
                    
                    # Add grid for better readability
                    ax.grid(True, alpha=0.3)
                    
                    # Add QL-CC-TMS mean values and CC-TMS comparison
                    for i, bus_val in enumerate(sorted(ql_proc_data['num_buses'].unique())):
                        # QL-CC-TMS data
                        ql_bus_data = ql_proc_data[ql_proc_data['num_buses'] == bus_val]['makespan']
                        # CC-TMS data
                        cc_bus_data = cc_proc_data[cc_proc_data['num_buses'] == bus_val]['makespan']
                        
                        if len(ql_bus_data) > 0:
                            ql_mean = ql_bus_data.mean()
                            ax.text(i, ql_mean, f'QL: {ql_mean:.1f}', 
                                   ha='center', va='bottom', fontweight='bold', color='red')
                            
                            # Add CC-TMS comparison line and annotation
                            if len(cc_bus_data) > 0:
                                cc_mean = cc_bus_data.mean()
                                # Draw horizontal line for CC-TMS makespan
                                ax.axhline(y=cc_mean, color='blue', linestyle='--', alpha=0.7, linewidth=2)
                                
                                # Add CC-TMS annotation
                                y_max = ax.get_ylim()[1]
                                ax.text(i, cc_mean + (y_max * 0.02), f'CC: {cc_mean:.1f}', 
                                       ha='center', va='bottom', fontweight='bold', color='blue')
                    
                    # Add legend
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        plt.Rectangle((0,0),1,1, facecolor='lightcoral', label='QL-CC-TMS'),
                        Line2D([0], [0], color='blue', linestyle='--', label='CC-TMS Reference')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right')
                    
                    # Adjust layout
                    plt.tight_layout()
                    
                    # Save plot
                    plot_filename = f"boxplot_{dag_type}_param_{param_value}_P{proc}_varying_buses_qlcctms_with_cctms.png"
                    plot_filepath = os.path.join(self.results_dir, 'plots', plot_filename)
                    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
                    
                    print(f"  Generated plot {plot_count}: {plot_filename}")
                    
                    # Close figure to free memory
                    plt.close(fig)
                
                # 2. CONSTANT BUSES, VARYING PROCESSORS
                for bus in self.buses:
                    # Only create plots for QL-CC-TMS, but annotate with CC-TMS comparison
                    plot_count += 1
                    
                    # Filter data for constant bus and QL-CC-TMS
                    ql_bus_data = dag_data[
                        (dag_data['num_buses'] == bus) & 
                        (dag_data['algorithm'] == 'qlcctms')
                    ]
                    
                    # Filter data for constant bus and CC-TMS (for comparison)
                    cc_bus_data = dag_data[
                        (dag_data['num_buses'] == bus) & 
                        (dag_data['algorithm'] == 'cctms')
                    ]
                    
                    if ql_bus_data.empty:
                        print(f"  Warning: No QL-CC-TMS data for B{bus}")
                        continue
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Create box plot with processors on X-axis for QL-CC-TMS
                    sns.boxplot(
                        data=ql_bus_data, 
                        x='num_processors', 
                        y='makespan',
                        ax=ax,
                        color='lightcoral'
                    )
                    
                    # Customize plot
                    title = f"{dag_type.replace('_', ' ').title()} (Param: {param_value}) - {bus} Bus{'es' if bus > 1 else ''} - QL-CC-TMS vs CC-TMS"
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    ax.set_xlabel('Number of Processors', fontsize=12)
                    ax.set_ylabel('Makespan (ms)', fontsize=12)
                    
                    # Add grid for better readability
                    ax.grid(True, alpha=0.3)
                    
                    # Add QL-CC-TMS mean values and CC-TMS comparison
                    for i, proc_val in enumerate(sorted(ql_bus_data['num_processors'].unique())):
                        # QL-CC-TMS data
                        ql_proc_data_subset = ql_bus_data[ql_bus_data['num_processors'] == proc_val]['makespan']
                        # CC-TMS data
                        cc_proc_data_subset = cc_bus_data[cc_bus_data['num_processors'] == proc_val]['makespan']
                        
                        if len(ql_proc_data_subset) > 0:
                            ql_mean = ql_proc_data_subset.mean()
                            ax.text(i, ql_mean, f'QL: {ql_mean:.1f}', 
                                   ha='center', va='bottom', fontweight='bold', color='red')
                            
                            # Add CC-TMS comparison line and annotation
                            if len(cc_proc_data_subset) > 0:
                                cc_mean = cc_proc_data_subset.mean()
                                # Draw horizontal line for CC-TMS makespan
                                ax.axhline(y=cc_mean, color='blue', linestyle='--', alpha=0.7, linewidth=2)
                                
                                # Add CC-TMS annotation
                                y_max = ax.get_ylim()[1]
                                ax.text(i, cc_mean + (y_max * 0.02), f'CC: {cc_mean:.1f}', 
                                       ha='center', va='bottom', fontweight='bold', color='blue')
                    
                    # Add legend
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        plt.Rectangle((0,0),1,1, facecolor='lightcoral', label='QL-CC-TMS'),
                        Line2D([0], [0], color='blue', linestyle='--', label='CC-TMS Reference')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right')
                    
                    # Adjust layout
                    plt.tight_layout()
                    
                    # Save plot
                    plot_filename = f"boxplot_{dag_type}_param_{param_value}_B{bus}_varying_processors_qlcctms_with_cctms.png"
                    plot_filepath = os.path.join(self.results_dir, 'plots', plot_filename)
                    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
                    
                    print(f"  Generated plot {plot_count}: {plot_filename}")
                    
                    # Close figure to free memory
                    plt.close(fig)
        
        print(f"\nAll {plot_count} box plots generated successfully!")
        print(f"Plots saved to: {os.path.join(self.results_dir, 'plots')}")
    
    def generate_line_plots(self, results_df):
        """
        Generate line plots showing average makespans for both algorithms.
        
        For each DAG type and parameter combination, generates:
        - Line plots with constant processors, varying buses
        - Line plots with constant buses, varying processors
        
        Args:
            results_df (pd.DataFrame): Experiment results
        """
        print(f"\n" + "="*80)
        print("GENERATING LINE PLOTS (AVERAGE MAKESPANS)")
        print("="*80)
        print("Generating line plots showing average makespans:")
        print("  - Constant processors, varying buses")
        print("  - Constant buses, varying processors")
        
        # Set up plotting style
        plt.style.use('default')
        
        plot_count = 0
        expected_plots = len(list(self.dag_configs.values())[0]) * (len(self.processors) + len(self.buses))
        
        # Generate plots for each DAG type and parameter
        for dag_type, param_values in self.dag_configs.items():
            for param_value in param_values:
                
                # Filter data for current DAG configuration
                dag_data = results_df[
                    (results_df['dag_type'] == dag_type) & 
                    (results_df['dag_parameter'] == param_value)
                ]
                
                if dag_data.empty:
                    print(f"Warning: No data for {dag_type} with parameter {param_value}")
                    continue
                
                print(f"\nProcessing {dag_type} (parameter: {param_value})...")
                
                # 1. CONSTANT PROCESSORS, VARYING BUSES
                for proc in self.processors:
                    plot_count += 1
                    
                    # Filter data for constant processor
                    proc_data = dag_data[dag_data['num_processors'] == proc]
                    
                    if proc_data.empty:
                        print(f"  Warning: No data for P{proc}")
                        continue
                    
                    # Calculate average makespans for each algorithm and bus count
                    avg_data = proc_data.groupby(['algorithm', 'num_buses'])['makespan'].mean().reset_index()
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot lines for each algorithm
                    for algorithm in self.algorithms:
                        alg_data = avg_data[avg_data['algorithm'] == algorithm]
                        if not alg_data.empty:
                            color = 'red' if algorithm == 'qlcctms' else 'blue'
                            marker = 'o' if algorithm == 'qlcctms' else 's'
                            label = 'QL-CC-TMS' if algorithm == 'qlcctms' else 'CC-TMS'
                            
                            ax.plot(alg_data['num_buses'], alg_data['makespan'], 
                                   color=color, marker=marker, linewidth=2, markersize=8,
                                   label=label)
                            
                            # Add value annotations
                            for _, row in alg_data.iterrows():
                                ax.annotate(f'{row["makespan"]:.1f}', 
                                          (row['num_buses'], row['makespan']),
                                          textcoords="offset points", xytext=(0,10), 
                                          ha='center', fontsize=9, color=color)
                    
                    # Customize plot
                    title = f"{dag_type.replace('_', ' ').title()} (Param: {param_value}) - {proc} Processors - Average Makespan"
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    ax.set_xlabel('Number of Buses', fontsize=12)
                    ax.set_ylabel('Average Makespan (ms)', fontsize=12)
                    
                    # Add grid and legend
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='best')
                    
                    # Set integer ticks for x-axis
                    ax.set_xticks(sorted(proc_data['num_buses'].unique()))
                    
                    # Adjust layout
                    plt.tight_layout()
                    
                    # Save plot
                    plot_filename = f"lineplot_{dag_type}_param_{param_value}_P{proc}_varying_buses_avg_makespan.png"
                    plot_filepath = os.path.join(self.results_dir, 'plots', plot_filename)
                    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
                    
                    print(f"  Generated line plot {plot_count}: {plot_filename}")
                    
                    # Close figure to free memory
                    plt.close(fig)
                
                # 2. CONSTANT BUSES, VARYING PROCESSORS
                for bus in self.buses:
                    plot_count += 1
                    
                    # Filter data for constant bus
                    bus_data = dag_data[dag_data['num_buses'] == bus]
                    
                    if bus_data.empty:
                        print(f"  Warning: No data for B{bus}")
                        continue
                    
                    # Calculate average makespans for each algorithm and processor count
                    avg_data = bus_data.groupby(['algorithm', 'num_processors'])['makespan'].mean().reset_index()
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot lines for each algorithm
                    for algorithm in self.algorithms:
                        alg_data = avg_data[avg_data['algorithm'] == algorithm]
                        if not alg_data.empty:
                            color = 'red' if algorithm == 'qlcctms' else 'blue'
                            marker = 'o' if algorithm == 'qlcctms' else 's'
                            label = 'QL-CC-TMS' if algorithm == 'qlcctms' else 'CC-TMS'
                            
                            ax.plot(alg_data['num_processors'], alg_data['makespan'], 
                                   color=color, marker=marker, linewidth=2, markersize=8,
                                   label=label)
                            
                            # Add value annotations
                            for _, row in alg_data.iterrows():
                                ax.annotate(f'{row["makespan"]:.1f}', 
                                          (row['num_processors'], row['makespan']),
                                          textcoords="offset points", xytext=(0,10), 
                                          ha='center', fontsize=9, color=color)
                    
                    # Customize plot
                    title = f"{dag_type.replace('_', ' ').title()} (Param: {param_value}) - {bus} Bus{'es' if bus > 1 else ''} - Average Makespan"
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    ax.set_xlabel('Number of Processors', fontsize=12)
                    ax.set_ylabel('Average Makespan (ms)', fontsize=12)
                    
                    # Add grid and legend
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='best')
                    
                    # Set integer ticks for x-axis
                    ax.set_xticks(sorted(bus_data['num_processors'].unique()))
                    
                    # Adjust layout
                    plt.tight_layout()
                    
                    # Save plot
                    plot_filename = f"lineplot_{dag_type}_param_{param_value}_B{bus}_varying_processors_avg_makespan.png"
                    plot_filepath = os.path.join(self.results_dir, 'plots', plot_filename)
                    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
                    
                    print(f"  Generated line plot {plot_count}: {plot_filename}")
                    
                    # Close figure to free memory
                    plt.close(fig)
        
        print(f"\nAll {plot_count} line plots generated successfully!")
        print(f"Line plots saved to: {os.path.join(self.results_dir, 'plots')}")
    
    def _add_statistical_annotations_by_category(self, ax, data, category_column):
        """
        Add statistical significance annotations to box plots grouped by category.
        
        Args:
            ax: Matplotlib axis object
            data (pd.DataFrame): Data for the current plot
            category_column (str): Column name to group by ('num_buses' or 'num_processors')
        """
        # Get unique category values (e.g., bus numbers or processor numbers)
        categories = sorted(data[category_column].unique())
        
        for i, category_val in enumerate(categories):
            category_data = data[data[category_column] == category_val]
            
            cctms_data = category_data[category_data['algorithm'] == 'cctms']['makespan']
            qlcctms_data = category_data[category_data['algorithm'] == 'qlcctms']['makespan']
            
            if len(cctms_data) > 0 and len(qlcctms_data) > 0:
                # Perform t-test (simple comparison)
                try:
                    t_stat, p_value = stats.ttest_ind(cctms_data, qlcctms_data)
                    
                    # Add significance marker if p < 0.05
                    if p_value < 0.05:
                        y_max = max(category_data['makespan'].max(), 
                                   cctms_data.max(), qlcctms_data.max())
                        ax.text(i, y_max * 1.05, '*', ha='center', va='bottom', 
                               fontsize=16, fontweight='bold')
                except:
                    pass  # Skip if statistical test fails
    
    def generate_summary_statistics(self, results_df):
        """
        Generate comprehensive summary statistics for the experiment.
        
        Args:
            results_df (pd.DataFrame): Experiment results
            
        Returns:
            dict: Summary statistics
        """
        print(f"\n" + "="*80)
        print("GENERATING SUMMARY STATISTICS")
        print("="*80)
        
        summary_stats = {}
        
        # Overall statistics
        summary_stats['overall'] = {
            'total_experiments': len(results_df),
            'unique_configurations': len(results_df.groupby(['dag_type', 'dag_parameter', 'platform_config', 'algorithm'])),
            'algorithms_tested': results_df['algorithm'].unique().tolist(),
            'dag_types_tested': results_df['dag_type'].unique().tolist()
        }
        
        # Algorithm comparison
        algorithm_stats = {}
        for algorithm in self.algorithms:
            alg_data = results_df[results_df['algorithm'] == algorithm]['makespan']
            if len(alg_data) > 0:
                algorithm_stats[algorithm] = {
                    'mean_makespan': float(alg_data.mean()),
                    'median_makespan': float(alg_data.median()),
                    'std_makespan': float(alg_data.std()),
                    'min_makespan': float(alg_data.min()),
                    'max_makespan': float(alg_data.max()),
                    'total_runs': len(alg_data)
                }
        
        summary_stats['algorithm_comparison'] = algorithm_stats
        
        # Print key findings
        print("\nKey Findings:")
        print(f"  Total experimental runs: {summary_stats['overall']['total_experiments']}")
        print(f"  Unique configurations: {summary_stats['overall']['unique_configurations']}")
        
        if len(algorithm_stats) == 2:
            cctms_mean = algorithm_stats[Algorithms.CCTMS]['mean_makespan']
            qlcctms_mean = algorithm_stats[Algorithms.QLCCTMS]['mean_makespan']
            
            print(f"\nAlgorithm Performance:")
            print(f"  CC-TMS average makespan: {cctms_mean:.2f} ms")
            print(f"  QL-CC-TMS average makespan: {qlcctms_mean:.2f} ms")
            
            if cctms_mean < qlcctms_mean:
                improvement = ((qlcctms_mean - cctms_mean) / qlcctms_mean) * 100
                print(f"  CC-TMS performs {improvement:.1f}% better on average")
            else:
                improvement = ((cctms_mean - qlcctms_mean) / cctms_mean) * 100
                print(f"  QL-CC-TMS performs {improvement:.1f}% better on average")
        
        return summary_stats
    
    def run_complete_experiment(self):
        """
        Run the complete absolute makespan comparison experiment.
        
        Returns:
            tuple: (results_dataframe, metadata, summary_statistics)
        """
        print("="*80)
        print("SMALL TEST - ABSOLUTE MAKESPAN COMPARISON EXPERIMENT")
        print("="*80)
        print("This small test will:")
        print(f"  - Test {len(self.dag_configs)} DAG type with {len(list(self.dag_configs.values())[0])} parameter values")
        print(f"  - Use {len(self.processors)} × {len(self.buses)} platform configurations")
        print(f"  - Compare {len(self.algorithms)} scheduling algorithms")
        print(f"  - Run {self.iterations} iterations per configuration")
        expected_plots = len(list(self.dag_configs.values())[0]) * (len(self.processors) + len(self.buses))
        print(f"  - Generate {expected_plots} detailed QL-CC-TMS box plots (with CC-TMS reference lines)")
        print(f"  - Generate {expected_plots} line plots showing average makespans (both algorithms)")
        print(f"  - Calculate comprehensive statistics")
        print("="*80)
        
        # Run the main experiment
        results_df, metadata = self.run_experiment()
        
        # Save results
        csv_file, json_file = self.save_results(results_df, metadata)
        
        # Generate visualizations
        self.generate_box_plots(results_df)
        self.generate_line_plots(results_df)
        
        # Generate summary statistics
        summary_stats = self.generate_summary_statistics(results_df)
        
        # Save summary statistics
        summary_file = os.path.join(self.results_dir, f"small_test_summary_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"\n" + "="*80)
        print("SMALL TEST EXPERIMENT COMPLETE!")
        print("="*80)
        print(f"Results files:")
        print(f"  - CSV data: {csv_file}")
        print(f"  - Metadata: {json_file}")
        print(f"  - Statistics: {summary_file}")
        expected_plots = len(list(self.dag_configs.values())[0]) * (len(self.processors) + len(self.buses))
        total_plots = expected_plots * 2  # Box plots + Line plots
        print(f"  - Plots: {os.path.join(self.results_dir, 'plots')} ({expected_plots} box plots + {expected_plots} line plots = {total_plots} total)")
        
        return results_df, metadata, summary_stats


def main():
    """
    Main function to run the small test for absolute makespan comparison experiment.
    """
    print("Starting Small Test - Absolute Makespan Comparison Experiment...")
    
    # Create experiment instance with reduced parameters for testing
    experiment = AbsoluteMakespanComparator(
        results_dir="./results/new_experiment_small_test", 
        iterations=30  # Reduced iterations for quick testing
    )
    
    # Run complete experiment
    results_df, metadata, summary_stats = experiment.run_complete_experiment()
    
    print("\nSmall test completed successfully!")
    print(f"Check the results directory for detailed outputs and visualizations.")
    print(f"If this works well, you can run the full experiment with more parameters and iterations.")


if __name__ == "__main__":
    main()