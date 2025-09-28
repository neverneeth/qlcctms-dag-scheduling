"""
Utility Functions for DAG Scheduling Experiments

This module provides common utility functions to reduce code duplication
across experiment files and provide consistent functionality.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import (
    DAGTypes, Algorithms, DEFAULT_QL_PARAMS, ExperimentConfig,
    PlotConfig, FileConfig, ProgressConfig, ErrorConfig, ValidationConfig
)


class ExperimentUtils:
    """Utility functions for experiment management."""
    
    @staticmethod
    def create_results_directory(results_dir: str) -> None:
        """Create results directory structure if it doesn't exist."""
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, FileConfig.PLOTS_SUBDIR), exist_ok=True)
    
    @staticmethod
    def save_experiment_results(results_df: pd.DataFrame, metadata: Dict[str, Any], 
                              results_dir: str, experiment_name: str) -> Tuple[str, str]:
        """Save experiment results and metadata to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV results
        csv_filename = FileConfig.CSV_PATTERN.format(
            experiment_name=experiment_name, 
            timestamp=timestamp
        )
        csv_filepath = os.path.join(results_dir, csv_filename)
        results_df.to_csv(csv_filepath, index=False)
        
        # Save JSON metadata
        json_filename = FileConfig.METADATA_PATTERN.format(
            experiment_name=experiment_name, 
            timestamp=timestamp
        )
        json_filepath = os.path.join(results_dir, json_filename)
        with open(json_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  CSV: {csv_filepath}")
        print(f"  Metadata: {json_filepath}")
        
        return csv_filepath, json_filepath
    
    @staticmethod
    def save_summary_statistics(summary_stats: Dict[str, Any], results_dir: str, 
                              experiment_name: str) -> str:
        """Save summary statistics to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = FileConfig.SUMMARY_PATTERN.format(
            experiment_name=experiment_name, 
            timestamp=timestamp
        )
        summary_filepath = os.path.join(results_dir, summary_filename)
        
        with open(summary_filepath, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        return summary_filepath
    
    @staticmethod
    def calculate_total_configurations(dag_configs: Dict[str, List], processors: List[int], 
                                     buses: List[int], algorithms: List[str], 
                                     iterations: int) -> int:
        """Calculate the total number of experimental configurations."""
        total = 0
        for dag_type, params in dag_configs.items():
            total += len(params) * len(processors) * len(buses) * len(algorithms) * iterations
        return total
    
    @staticmethod
    def should_report_progress(current_config: int, total_configs: int, 
                             is_small_test: bool = False) -> bool:
        """Determine if progress should be reported for current configuration."""
        interval = ProgressConfig.PROGRESS_INTERVAL_SMALL if is_small_test else ProgressConfig.PROGRESS_INTERVAL_FULL
        return current_config % interval == 0 or current_config == total_configs


class DAGUtils:
    """Utility functions for DAG generation and parameter handling."""
    
    @staticmethod
    def generate_dag_with_parameters(dag_generator, dag_type: str, param_value: int):
        """Generate DAG with appropriate parameters based on type."""
        try:
            if dag_type == DAGTypes.GAUSSIAN:
                return dag_generator.generate(chi=param_value)
            elif dag_type == DAGTypes.EPIGENOMICS:
                return dag_generator.generate(gamma=param_value)
            elif dag_type == DAGTypes.LAPLACE:
                return dag_generator.generate(phi=param_value)
            elif dag_type == DAGTypes.STENCIL:
                # Fixed: only pass xi parameter
                return dag_generator.generate(xi=param_value)
            else:
                raise ValueError(f"Unknown DAG type: {dag_type}")
        except Exception as e:
            print(f"Error generating {dag_type} DAG with parameter {param_value}: {e}")
            raise
    
    @staticmethod
    def validate_dag_result(dag, task_list: List[str], message_list: List[str], 
                           dag_type: str, param_value: int) -> bool:
        """Validate that DAG generation was successful."""
        if dag is None or not task_list:
            print(f"Warning: {dag_type} DAG generation failed for parameter {param_value}")
            return False
        
        if len(task_list) == 0:
            print(f"Warning: {dag_type} DAG has no tasks for parameter {param_value}")
            return False
        
        return True


class SchedulerUtils:
    """Utility functions for scheduler creation and management."""
    
    @staticmethod
    def create_scheduler_with_params(algorithm: str, ql_params: Dict[str, Any] = None):
        """Create scheduler with appropriate parameters."""
        from src.schedulers import SchedulerFactory
        
        if algorithm == Algorithms.QLCCTMS:
            params = ql_params or DEFAULT_QL_PARAMS
            return SchedulerFactory.create_scheduler(algorithm, **params)
        else:
            return SchedulerFactory.create_scheduler(algorithm)
    
    @staticmethod
    def execute_scheduling_with_retry(scheduler, dag, task_list: List[str], 
                                    message_list: List[str], ET: np.ndarray, 
                                    CT: np.ndarray, random_state: int = None,
                                    max_retries: int = None) -> Optional[Dict[str, Any]]:
        """Execute scheduling with retry logic for robustness."""
        max_retries = max_retries or ErrorConfig.MAX_RETRIES
        
        for attempt in range(max_retries + 1):
            try:
                result = scheduler.schedule(
                    dag, task_list, message_list, ET, CT,
                    random_state=random_state + attempt if random_state is not None else None
                )
                return result
            except Exception as e:
                if attempt < max_retries:
                    print(f"        Scheduling attempt {attempt + 1} failed: {e}. Retrying...")
                    continue
                else:
                    print(f"        Scheduling failed after {max_retries + 1} attempts: {e}")
                    return None
        
        return None


class StatisticsUtils:
    """Utility functions for statistical analysis and summary generation."""
    
    @staticmethod
    def generate_algorithm_comparison(results_df: pd.DataFrame, 
                                    algorithms: List[str]) -> Dict[str, Dict[str, float]]:
        """Generate algorithm comparison statistics."""
        algorithm_stats = {}
        for algorithm in algorithms:
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
        return algorithm_stats
    
    @staticmethod
    def print_algorithm_performance(algorithm_stats: Dict[str, Dict[str, float]]) -> None:
        """Print algorithm performance comparison."""
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


class PlotUtils:
    """Utility functions for consistent plotting across experiments."""
    
    @staticmethod
    def setup_plot_style() -> None:
        """Set up consistent plotting style."""
        plt.style.use('default')
        sns.set_palette("husl")
    
    @staticmethod
    def get_algorithm_plot_properties(algorithm: str) -> Tuple[str, str, str]:
        """Get consistent color, marker, and label for algorithm."""
        if algorithm == Algorithms.QLCCTMS:
            return PlotConfig.QLCCTMS_COLOR, PlotConfig.QLCCTMS_MARKER, PlotConfig.QLCCTMS_LABEL
        else:
            return PlotConfig.CCTMS_COLOR, PlotConfig.CCTMS_MARKER, PlotConfig.CCTMS_LABEL
    
    @staticmethod
    def save_plot(fig, results_dir: str, filename: str) -> str:
        """Save plot with consistent settings."""
        filepath = os.path.join(results_dir, FileConfig.PLOTS_SUBDIR, filename)
        fig.savefig(filepath, dpi=PlotConfig.DPI, bbox_inches='tight', format=PlotConfig.FORMAT)
        return filepath
    
    @staticmethod
    def close_plot(fig) -> None:
        """Close plot to free memory."""
        plt.close(fig)


class ValidationUtils:
    """Utility functions for input validation and error checking."""
    
    @staticmethod
    def validate_experiment_parameters(dag_configs: Dict[str, List], processors: List[int], 
                                     buses: List[int], iterations: int) -> bool:
        """Validate experiment parameters."""
        # Validate DAG configs
        for dag_type, params in dag_configs.items():
            if not DAGTypes.validate_type(dag_type):
                print(f"Error: Invalid DAG type '{dag_type}'")
                return False
            if not params or len(params) == 0:
                print(f"Error: No parameters specified for DAG type '{dag_type}'")
                return False
        
        # Validate processors
        if not processors or len(processors) == 0:
            print("Error: No processors specified")
            return False
        for p in processors:
            if p < ValidationConfig.MIN_PROCESSORS or p > ValidationConfig.MAX_PROCESSORS:
                print(f"Error: Invalid processor count {p}")
                return False
        
        # Validate buses  
        if not buses or len(buses) == 0:
            print("Error: No buses specified")
            return False
        for b in buses:
            if b < ValidationConfig.MIN_BUSES or b > ValidationConfig.MAX_BUSES:
                print(f"Error: Invalid bus count {b}")
                return False
        
        # Validate iterations
        if iterations < ValidationConfig.MIN_ITERATIONS or iterations > ValidationConfig.MAX_ITERATIONS:
            print(f"Error: Invalid iteration count {iterations}")
            return False
        
        return True
    
    @staticmethod
    def check_file_permissions(results_dir: str) -> bool:
        """Check if we can write to the results directory."""
        try:
            test_file = os.path.join(results_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return True
        except Exception as e:
            print(f"Error: Cannot write to results directory '{results_dir}': {e}")
            return False