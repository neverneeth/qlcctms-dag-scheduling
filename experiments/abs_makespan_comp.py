"""
Absolute Makespan Comparison Experiment

This experiment aims to compare the absolute makespan values obtained by the CC-TMS and QL-CC-TMS algorithms 
For all four DAG configurations with a set of parameters (Gaussian Elimination with χ={3, 4, 5, 6}, Epigenomics with γ={2, 3, 4, 5}, 
Laplace with φ={2, 3, 4, 5}, Stencil with ξ={2, 3, 4, 5}) across various platform settings, p = {2, 4, 6, 8}
and b = {1, 2, 3, 4} and constant CCR = 1.0. 

The results are saved to a CSV file for further analysis.

The experiment runs 100 iterations for each configuration to ensure statistical significance.

The experiment generates 32 box plots. In each plot the X axis represents the platform settings,
while the Y axis represents the absolute makespan values for the CC-TMS and QL-CC-TMS algorithms. Each plot
corresponds to a specific DAG configuration and parameter setting.

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

# Add the parent directory to the path to import framework modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dag_generators import DAGFactory
from src.schedulers import SchedulerFactory
from src.cost_matrices import CostMatrixGenerator
from src.experiment_runner import ExperimentRunner, ExperimentConfig, ExperimentResult


class AbsoluteMakespanComparator:
    """
    Class to handle the absolute makespan comparison experiment.
    
    This experiment compares CC-TMS and QL-CC-TMS algorithms across:
    - 4 DAG types with different parameters
    - Various platform configurations (processors × buses)
    - 100 iterations per configuration for statistical significance
    - Generates comprehensive box plots for visualization
    """
    
    def __init__(self, results_dir="./results", iterations=100):
        """
        Initialize the experiment with configuration parameters.
        
        Args:
            results_dir (str): Directory to save results
            iterations (int): Number of iterations per configuration
        """
        self.results_dir = results_dir
        self.iterations = iterations
        self.ccr = 1.0  # Constant CCR value as specified
        
        # Experiment parameters as specified in the requirements
        self.dag_configs = {
            'gaussian_elimination': [3, 4, 5, 6],      # χ values
            'epigenomics': [2, 3, 4, 5],               # γ values  
            'laplace': [2, 3, 4, 5],                   # φ values
            'stencil': [2, 3, 4, 5]                    # ξ values (using ξ for width, λ for height)
        }
        
        # Platform configurations
        self.processors = [2, 4, 6, 8]                 # p values
        self.buses = [1, 2, 3, 4]                      # b values
        
        # Algorithms to compare
        self.algorithms = ['cctms', 'qlcctms']
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'plots'), exist_ok=True)
        
        print(f"Absolute Makespan Comparison Experiment Initialized")
        print(f"Results directory: {self.results_dir}")
        print(f"Iterations per configuration: {self.iterations}")
        print(f"Total configurations: {self._calculate_total_configurations()}")
    
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
                if dag_type == 'gaussian_elimination':
                    dag, task_list, message_list = dag_generator.generate(chi=param_value)
                elif dag_type == 'epigenomics':
                    dag, task_list, message_list = dag_generator.generate(gamma=param_value)
                elif dag_type == 'laplace':
                    dag, task_list, message_list = dag_generator.generate(phi=param_value)
                elif dag_type == 'stencil':
                    # For stencil, use param_value for both width and height for simplicity
                    dag, task_list, message_list = dag_generator.generate(lambda_val=param_value, xi=param_value)
                
                # Test across all platform configurations
                for num_proc, num_bus in product(self.processors, self.buses):
                    platform_config = f"P{num_proc}B{num_bus}"
                    print(f"    Platform: {platform_config}")
                    
                    # Generate cost matrices for current platform
                    cost_gen = CostMatrixGenerator()
                    ET, CT = cost_gen.generate_cost_matrices(
                        len(task_list), len(message_list), 
                        num_proc, num_bus, self.ccr
                    )
                    
                    # Test both algorithms
                    for algorithm in self.algorithms:
                        print(f"      Algorithm: {algorithm.upper()}")
                        
                        # Create scheduler
                        if algorithm == 'qlcctms':
                            scheduler = SchedulerFactory.create_scheduler(algorithm, max_episodes=5000)
                        else:
                            scheduler = SchedulerFactory.create_scheduler(algorithm)
                        
                        # Run multiple iterations for statistical significance
                        iteration_results = []
                        for iteration in range(self.iterations):
                            current_config += 1
                            
                            # Show progress
                            if current_config % 100 == 0 or current_config == total_configs:
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
        Generate 32 box plots comparing CC-TMS and QL-CC-TMS makespan values.
        
        Each plot corresponds to a specific DAG type and parameter value,
        showing makespan distributions across different platform configurations.
        
        Args:
            results_df (pd.DataFrame): Experiment results
        """
        print(f"\n" + "="*80)
        print("GENERATING BOX PLOTS")
        print("="*80)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        plot_count = 0
        
        # Generate plots for each DAG type and parameter
        for dag_type, param_values in self.dag_configs.items():
            for param_value in param_values:
                plot_count += 1
                
                # Filter data for current DAG configuration
                dag_data = results_df[
                    (results_df['dag_type'] == dag_type) & 
                    (results_df['dag_parameter'] == param_value)
                ]
                
                if dag_data.empty:
                    print(f"Warning: No data for {dag_type} with parameter {param_value}")
                    continue
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Create box plot
                sns.boxplot(
                    data=dag_data, 
                    x='platform_config', 
                    y='makespan', 
                    hue='algorithm',
                    ax=ax
                )
                
                # Customize plot
                title = f"{dag_type.replace('_', ' ').title()} (Parameter: {param_value})"
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel('Platform Configuration (Processors × Buses)', fontsize=12)
                ax.set_ylabel('Makespan (ms)', fontsize=12)
                
                # Customize legend
                legend = ax.legend(title='Algorithm', loc='upper right')
                legend.get_title().set_fontweight('bold')
                
                # Add grid for better readability
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels if needed
                plt.xticks(rotation=45)
                
                # Add statistical annotations
                self._add_statistical_annotations(ax, dag_data)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save plot
                plot_filename = f"boxplot_{dag_type}_param_{param_value}.png"
                plot_filepath = os.path.join(self.results_dir, 'plots', plot_filename)
                plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
                
                print(f"Generated plot {plot_count}/32: {plot_filename}")
                
                # Close figure to free memory
                plt.close(fig)
        
        print(f"\nAll {plot_count} box plots generated successfully!")
        print(f"Plots saved to: {os.path.join(self.results_dir, 'plots')}")
    
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
        
        This method orchestrates the entire experimental process:
        1. Runs all experimental configurations
        2. Saves results to CSV and metadata to JSON
        3. Generates 32 box plots for visualization
        4. Computes summary statistics
        
        Returns:
            tuple: (results_dataframe, metadata, summary_statistics)
        """
        print("="*80)
        print("ABSOLUTE MAKESPAN COMPARISON EXPERIMENT")
        print("="*80)
        print("This experiment will:")
        print(f"  - Test {len(self.dag_configs)} DAG types with different parameters")
        print(f"  - Use {len(self.processors)} × {len(self.buses)} platform configurations")
        print(f"  - Compare {len(self.algorithms)} scheduling algorithms")
        print(f"  - Run {self.iterations} iterations per configuration")
        print(f"  - Generate 32 detailed box plots")
        print(f"  - Calculate comprehensive statistics")
        print("="*80)
        
        # Run the main experiment
        results_df, metadata = self.run_experiment()
        
        # Save results
        csv_file, json_file = self.save_results(results_df, metadata)
        
        # Generate visualizations
        self.generate_box_plots(results_df)
        
        # Generate summary statistics
        summary_stats = self.generate_summary_statistics(results_df)
        
        # Save summary statistics
        summary_file = os.path.join(self.results_dir, f"summary_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"\n" + "="*80)
        print("EXPERIMENT COMPLETE!")
        print("="*80)
        print(f"Results files:")
        print(f"  - CSV data: {csv_file}")
        print(f"  - Metadata: {json_file}")
        print(f"  - Statistics: {summary_file}")
        print(f"  - Box plots: {os.path.join(self.results_dir, 'plots')} (32 plots)")
        
        return results_df, metadata, summary_stats


def main():
    """
    Main function to run the absolute makespan comparison experiment.
    """
    print("Starting Absolute Makespan Comparison Experiment...")
    
    # Create experiment instance
    experiment = AbsoluteMakespanComparator(
        results_dir="./results", 
        iterations=100  # 100 iterations as specified
    )
    
    # Run complete experiment
    results_df, metadata, summary_stats = experiment.run_complete_experiment()
    
    print("\nExperiment completed successfully!")
    print(f"Check the results directory for detailed outputs and visualizations.")


if __name__ == "__main__":
    main()