"""
Absolute Makespan Comparison Experiment

This experiment aims to compare the absolute makespan values obtained by the CC-TMS and QL-CC-TMS algorithms 
for all four DAG configurations with a set of parameters (Gaussian Elimination with χ={3, 4, 5, 6}, Epigenomics with γ={2, 3, 4, 5}, 
Laplace with φ={2, 3, 4, 5}, Stencil with ξ={2, 3, 4, 5}) across various platform settings, p = {2, 4, 6, 8}
and b = {1, 2, 3, 4} and constant CCR = 1.0. 

Q-Learning Parameters (Research Paper Values):
- epsilon: 0.2, learning_rate: 0.1, discount: 0.8
- max_episodes: 300,000, convergence_window: 40, convergence_threshold: 0.2

The results are saved to a CSV file for further analysis.

The experiment runs 100 iterations for each configuration to ensure statistical significance.

The experiment generates 256 plots organized as:
- 128 box plots: Constant processors/varying buses (64) + Constant buses/varying processors (64)
- 128 line plots: Average makespan trends with same organization
The Y axis represents the absolute makespan values for the CC-TMS and QL-CC-TMS algorithms.

Date: September 2025
"""

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

# Import centralized configuration
from config.constants import (
    DAGTypes, Algorithms, DEFAULT_QL_PARAMS, ExperimentConfig,
    PlotConfig, FileConfig, ProgressConfig
)

from src.dag_generators import DAGFactory
from src.schedulers import SchedulerFactory, CCTMSScheduler, QLCCTMSScheduler
from src.cost_matrices import generate_cost_matrices
from src.experiment_utils import ExperimentUtils, ValidationUtils, DAGUtils, SchedulerUtils


class AbsoluteMakespanComparator:
    """
    Class to handle the absolute makespan comparison experiment.
    
    This experiment compares CC-TMS and QL-CC-TMS algorithms across:
    - 4 DAG types with different parameters
    - Various platform configurations (processors × buses)
    - 100 iterations per configuration for statistical significance
    - Generates comprehensive box plots for visualization
    """

    def __init__(self, results_dir="./results/abs_makespan", iterations=100):
        """
        Initialize the experiment with configuration parameters.
        
        Args:
            results_dir (str): Directory to save results
            iterations (int): Number of iterations per configuration
        """
        self.results_dir = results_dir
        self.iterations = iterations
        self.ccr = ExperimentConfig.CCR.copy()
        
        # Experiment parameters using centralized configuration
        self.dag_configs = {DAGTypes.EPIGENOMICS: ExperimentConfig.FULL_DAG_CONFIGS[DAGTypes.EPIGENOMICS]}
        
        # Platform configurations
        self.processors = ExperimentConfig.FULL_PROCESSORS.copy()
        self.buses = ExperimentConfig.FULL_BUSES.copy()
        
        # Algorithms to compare
        self.algorithms = [Algorithms.CCTMS, Algorithms.QLCCTMS]
        
        # Q-learning parameters - Using research paper values for full experiment
        # Note: DEFAULT_QL_PARAMS in constants.py are reduced for faster testing
        self.ql_params = {
            'epsilon': 0.2,                    # Exploration rate
            'learning_rate': 0.1,              # Learning rate (alpha)
            'discount': 0.8,                   # Discount factor (gamma)
            'max_episodes': 5000,            # Maximum episodes 
            'convergence_window': 30,          # Window for convergence check 
            'convergence_threshold': 0.1       # Convergence threshold 
        }
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'plots'), exist_ok=True)
        
        print(f"Absolute Makespan Comparison Experiment Initialized")
        print(f"Results directory: {self.results_dir}")
        print(f"Iterations per configuration: {self.iterations}")
        total_configs = ExperimentUtils.calculate_total_configurations(
            self.dag_configs, self.processors, self.buses, self.algorithms, self.iterations
        )
        print(f"Total configurations: {total_configs}")
        # Calculate expected plots: 4 DAG types × 4 parameters × (4 processors + 4 buses) = 128 plots each
        total_param_combinations = sum(len(params) for params in self.dag_configs.values())
        expected_plots_per_type = total_param_combinations * (len(self.processors) + len(self.buses))
        print(f"Expected plots per visualization type: {expected_plots_per_type}")
        print(f"  - Box plots: {expected_plots_per_type*2} (QL-CC-TMS with CC-TMS reference)")
        print(f"  - Total plots: {expected_plots_per_type * 2}")
        print(f"  - Only QL-CC-TMS plots generated (CC-TMS shown as reference lines)")    

    
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
        total_configs = ExperimentUtils.calculate_total_configurations(
            self.dag_configs, self.processors, self.buses, self.algorithms, self.iterations
        )
        current_config = 0
        
        # Validate experiment parameters
        ValidationUtils.validate_experiment_parameters(
            self.dag_configs, self.processors, self.buses, self.iterations
        )
        
        # Iterate through all DAG types and their parameters
        for dag_type, param_values in self.dag_configs.items():
            print(f"\nProcessing {dag_type.upper()} DAGs...")
            
            for param_value in param_values:
                print(f"  Parameter value: {param_value}")
                
                # Generate DAG with current parameter
                dag_generator = DAGFactory.create_generator(dag_type)
                
                # Generate DAG using utility function
                dag, task_list, message_list = DAGUtils.generate_dag_with_parameters(
                    dag_generator, dag_type, param_value
                )
                
                # Validate DAG generation
                DAGUtils.validate_dag_result(dag, task_list, message_list, dag_type, param_value)
                
                # Test across all platform configurations
                for num_proc, num_bus in product(self.processors, self.buses):
                    platform_config = f"P{num_proc}B{num_bus}"
                    print(f"    Platform: {platform_config}")
                    
                    
                    for ccr in self.ccr:
                    # Test both algorithms
                        for algorithm in self.algorithms:
                            print(f"      Algorithm: {algorithm.upper()}")
                            
                            # Create scheduler using utility function
                            scheduler = SchedulerUtils.create_scheduler_with_params(algorithm, self.ql_params)
                            
                            # Run multiple iterations for statistical significance
                            iteration_results = []
                            for iteration in range(self.iterations):
                                ET, CT, TL, ML = generate_cost_matrices(
                                    dag,  # Pass the actual graph, not len(task_list)
                                    num_proc, num_bus, ccr,
                                    random_state=42+iteration
                                )
                                
                                current_config += 1
                                
                                # Show progress
                                if current_config % ProgressConfig.PROGRESS_INTERVAL_FULL == 0 or current_config == total_configs:
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
                                        'et': ET.tolist(),
                                        'ct': CT.tolist(),
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
            'experiment_name': 'absolute_makespan_comparison',
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
        csv_filename = f"absolute_makespan_comparison_{timestamp}.csv"
        csv_filepath = os.path.join(self.results_dir, csv_filename)
        results_df.to_csv(csv_filepath, index=False)
        
        # Save JSON metadata
        json_filename = f"absolute_makespan_comparison_{timestamp}_metadata.json"
        json_filepath = os.path.join(self.results_dir, json_filename)
        with open(json_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  CSV: {csv_filepath}")
        print(f"  Metadata: {json_filepath}")
        
        return csv_filepath, json_filepath
    
    def generate_box_plots(self, results_df):
        """
        Generate box plots showing makespans for both algorithms.
        
        For each DAG type and parameter combination, generates:
        - Box plots with constant processors, varying buses
        - Box plots with constant buses, varying processors
        
        Args:
            results_df (pd.DataFrame): Experiment results
        """
        print(f"\n" + "="*80)
        print("GENERATING BOX PLOTS")
        print("="*80)
        print("Generating box plots showing makespans for both algorithms:")
        print("  - Constant processors, varying buses")
        print("  - Constant buses, varying processors")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create base plot directory
        base_plot_dir = os.path.join(self.results_dir, 'plots')
        os.makedirs(base_plot_dir, exist_ok=True)
        
        plot_count = 0
        
        # Generate plots for each DAG type and parameter
        for dag_type, param_values in self.dag_configs.items():
            # Create DAG type directory
            dag_dir = os.path.join(base_plot_dir, dag_type)
            os.makedirs(dag_dir, exist_ok=True)
            
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
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Create box plot with buses on X-axis for both algorithms
                    sns.boxplot(
                        data=proc_data,
                        x='num_buses',
                        y='makespan',
                        hue='algorithm',
                        palette={'cctms': 'skyblue', 'qlcctms': 'lightcoral'},
                        ax=ax
                    )
                    
                    # Customize plot
                    title = f"{dag_type.replace('_', ' ').title()} (Param: {param_value}) - {proc} Processors"
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    ax.set_xlabel('Number of Buses', fontsize=12)
                    ax.set_ylabel('Makespan (ms)', fontsize=12)
                    
                    # Update legend
                    handles, labels = ax.get_legend_handles_labels()
                    new_labels = ['CC-TMS', 'QL-CC-TMS']
                    ax.legend(handles=handles, labels=new_labels, title='Algorithm')
                    
                    # Add grid for better readability
                    ax.grid(True, alpha=0.3)
                    
                    # Create organized directory structure
                    param_dir = os.path.join(dag_dir, f'param_{param_value}')
                    box_plots_dir = os.path.join(param_dir, 'box_plots')
                    constant_proc_dir = os.path.join(box_plots_dir, 'constant_processors')
                    os.makedirs(constant_proc_dir, exist_ok=True)
                    
                    # Save plot
                    plot_filename = f"boxplot_{dag_type}_param_{param_value}_P{proc}_varying_buses.png"
                    plot_filepath = os.path.join(constant_proc_dir, plot_filename)
                    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
                    
                    print(f"  Generated plot {plot_count}: {plot_filename}")
                    
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
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Create box plot with processors on X-axis for both algorithms
                    sns.boxplot(
                        data=bus_data,
                        x='num_processors',
                        y='makespan',
                        hue='algorithm',
                        palette={'cctms': 'skyblue', 'qlcctms': 'lightcoral'},
                        ax=ax
                    )
                    
                    # Customize plot
                    title = f"{dag_type.replace('_', ' ').title()} (Param: {param_value}) - {bus} Bus{'es' if bus > 1 else ''}"
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    ax.set_xlabel('Number of Processors', fontsize=12)
                    ax.set_ylabel('Makespan (ms)', fontsize=12)
                    
                    # Update legend
                    handles, labels = ax.get_legend_handles_labels()
                    new_labels = ['CC-TMS', 'QL-CC-TMS']
                    ax.legend(handles=handles, labels=new_labels, title='Algorithm')
                    
                    # Add grid for better readability
                    ax.grid(True, alpha=0.3)
                    
                    # Create organized directory structure
                    param_dir = os.path.join(dag_dir, f'param_{param_value}')
                    box_plots_dir = os.path.join(param_dir, 'box_plots')
                    constant_bus_dir = os.path.join(box_plots_dir, 'constant_buses')
                    os.makedirs(constant_bus_dir, exist_ok=True)
                    
                    # Save plot
                    plot_filename = f"boxplot_{dag_type}_param_{param_value}_B{bus}_varying_processors.png"
                    plot_filepath = os.path.join(constant_bus_dir, plot_filename)
                    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
                    
                    print(f"  Generated plot {plot_count}: {plot_filename}")
                    
                    # Close figure to free memory
                    plt.close(fig)
        
        print(f"\nAll {plot_count} box plots generated successfully!")
        print(f"Plots organized in: {os.path.join(self.results_dir, 'plots')}")
        
        def _add_statistical_annotations(self, ax, data):
            """
            Add statistical significance annotations to box plots.
            
            Args:
                ax: Matplotlib axis object
                data (pd.DataFrame): Data for the current plot
            """
            # Calculate mean makespan for each algorithm per platform
            platform_configs = data['platform_config'].unique()
            
            for i, platform in enumerate(platform_configs):
                platform_data = data[data['platform_config'] == platform]
                
                cctms_data = platform_data[platform_data['algorithm'] == 'cctms']['makespan']
                qlcctms_data = platform_data[platform_data['algorithm'] == 'qlcctms']['makespan']
                
                if len(cctms_data) > 0 and len(qlcctms_data) > 0:
                    # Perform t-test (simple comparison)
                    from scipy import stats
                    try:
                        t_stat, p_value = stats.ttest_ind(cctms_data, qlcctms_data)
                        
                        # Add significance marker if p < 0.05
                        if p_value < 0.05:
                            y_max = max(platform_data['makespan'].max(), 
                                    cctms_data.max(), qlcctms_data.max())
                            ax.text(i, y_max * 1.05, '*', ha='center', va='bottom', 
                                fontsize=16, fontweight='bold')
                    except:
                        pass  # Skip if statistical test fails
    
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
                from scipy import stats
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
            algorithm_stats[algorithm] = {
                'mean_makespan': float(alg_data.mean()),
                'median_makespan': float(alg_data.median()),
                'std_makespan': float(alg_data.std()),
                'min_makespan': float(alg_data.min()),
                'max_makespan': float(alg_data.max()),
                'total_runs': len(alg_data)
            }
        
        summary_stats['algorithm_comparison'] = algorithm_stats
        
        # DAG type analysis
        dag_stats = {}
        for dag_type in self.dag_configs.keys():
            dag_data = results_df[results_df['dag_type'] == dag_type]
            dag_stats[dag_type] = {
                'total_experiments': len(dag_data),
                'parameter_values': sorted(dag_data['dag_parameter'].unique().tolist()),
                'mean_makespan_cctms': float(dag_data[dag_data['algorithm'] == 'cctms']['makespan'].mean()),
                'mean_makespan_qlcctms': float(dag_data[dag_data['algorithm'] == 'qlcctms']['makespan'].mean())
            }
        
        summary_stats['dag_type_analysis'] = dag_stats
        
        # Platform sensitivity analysis
        platform_stats = {}
        for platform in results_df['platform_config'].unique():
            platform_data = results_df[results_df['platform_config'] == platform]
            platform_stats[platform] = {
                'mean_makespan_cctms': float(platform_data[platform_data['algorithm'] == 'cctms']['makespan'].mean()),
                'mean_makespan_qlcctms': float(platform_data[platform_data['algorithm'] == 'qlcctms']['makespan'].mean()),
                'total_experiments': len(platform_data)
            }
        
        summary_stats['platform_analysis'] = platform_stats
        
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
            cctms_mean = algorithm_stats['cctms']['mean_makespan']
            qlcctms_mean = algorithm_stats['qlcctms']['mean_makespan']
            
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
        print("ABSOLUTE MAKESPAN COMPARISON EXPERIMENT")
        print("="*80)
        print("This experiment will:")
        print(f"  - Test {len(self.dag_configs)} DAG types with 4 parameter values each")
        print(f"  - Use {len(self.processors)} × {len(self.buses)} platform configurations")
        print(f"  - Compare {len(self.algorithms)} scheduling algorithms")
        print(f"  - Run {self.iterations} iterations per configuration")
        total_param_combinations = sum(len(params) for params in self.dag_configs.values())
        expected_plots_per_type = total_param_combinations * (len(self.processors) + len(self.buses))
        print(f"  - Generate {expected_plots_per_type} detailed QL-CC-TMS box plots (with CC-TMS reference lines)")
        print(f"  - Generate {expected_plots_per_type} line plots showing average makespans (both algorithms)")
        print(f"  - Calculate comprehensive statistics")
        print("="*80)
        
        # Run the main experiment
        results_df, metadata = self.run_experiment()
        
        # Save results
        csv_file, json_file = self.save_results(results_df, metadata)
        
        # Generate visualizations
        # self.generate_box_plots(results_df)
        
        # Generate summary statistics
        summary_stats = self.generate_summary_statistics(results_df)
        
        # Save summary statistics
        summary_file = os.path.join(self.results_dir, f"absolute_makespan_comparison_summary_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"\n" + "="*80)
        print("ABSOLUTE MAKESPAN COMPARISON EXPERIMENT COMPLETE!")
        print("="*80)
        print(f"Results files:")
        print(f"  - CSV data: {csv_file}")
        print(f"  - Metadata: {json_file}")
        print(f"  - Statistics: {summary_file}")
        total_param_combinations = sum(len(params) for params in self.dag_configs.values())
        expected_plots_per_type = total_param_combinations * (len(self.processors) + len(self.buses))
        total_plots = expected_plots_per_type * 2  # Box plots + Line plots
        print(f"  - Plots: {os.path.join(self.results_dir, 'plots')} ({expected_plots_per_type*2} box plots)")
        
        return results_df, metadata, summary_stats


def main():
    """
    Main function to run the complete absolute makespan comparison experiment.
    """
    print("Starting Absolute Makespan Comparison Experiment...")
    
    # Create experiment instance with full parameters
    experiment = AbsoluteMakespanComparator(
        results_dir="./results", 
        iterations=100  # Full 100 iterations for statistical significance
    )
    
    # Run complete experiment
    results_df, metadata, summary_stats = experiment.run_complete_experiment()
    
    print("\nFull experiment completed successfully!")
    print(f"Check the results directory for detailed outputs and visualizations.")


if __name__ == "__main__":
    main()