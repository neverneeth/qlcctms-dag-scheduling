"""
Experiment Runner Module

Runs experiments based on configurations and collects results.
"""

import os
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from .dag_generators import DAGFactory
from .schedulers import SchedulerFactory
from .cost_matrices import generate_cost_matrices


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    dag_type: str
    dag_params: Dict[str, Any]
    processors: List[int]
    buses: List[int]
    ccr_values: List[float]
    algorithms: List[str]
    num_runs: int = 100
    et_min: float = 10.0
    et_max: float = 30.0
    ct_min: float = 10.0
    ct_max: float = 30.0
    random_state_base: int = 42


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    dag_type: str
    dag_param_value: Any
    num_processors: int
    num_buses: int
    ccr: float
    algorithm: str
    run_id: int
    makespan: float
    execution_time: float
    num_tasks: int
    num_messages: int
    converged: Optional[bool] = None
    episodes: Optional[int] = None


class ExperimentRunner:
    """Runs scheduling experiments and collects results."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def run_experiment(self, config: ExperimentConfig, 
                      progress_callback: Optional[callable] = None) -> List[ExperimentResult]:
        """
        Run a complete experiment based on configuration.
        
        Args:
            config: Experiment configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of experiment results
        """
        results = []
        total_runs = self._calculate_total_runs(config)
        current_run = 0
        
        # Create DAG generator
        dag_generator = DAGFactory.create_generator(config.dag_type)
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(config)
        
        for dag_param_value, num_proc, num_bus, ccr, algorithm in param_combinations:
            for run_id in range(config.num_runs):
                current_run += 1
                
                if progress_callback:
                    progress_callback(current_run, total_runs)
                
                # Generate DAG
                if config.dag_type == 'gaussian':
                    graph, task_list, message_list = dag_generator.generate(chi=dag_param_value)
                elif config.dag_type == 'epigenomics':
                    graph, task_list, message_list = dag_generator.generate(gamma=dag_param_value)
                elif config.dag_type == 'laplace':
                    graph, task_list, message_list = dag_generator.generate(phi=dag_param_value)
                elif config.dag_type == 'stencil':
                    graph, task_list, message_list = dag_generator.generate(xi=dag_param_value)
                else:
                    raise ValueError(f"Unknown DAG type: {config.dag_type}")
                
                # Generate cost matrices
                random_state = config.random_state_base + run_id
                ET, CT, _, _ = generate_cost_matrices(
                    graph, num_proc, num_bus, ccr,
                    config.et_min, config.et_max,
                    config.ct_min, config.ct_max,
                    random_state
                )
                
                # Run scheduling algorithm
                scheduler = SchedulerFactory.create_scheduler(algorithm)
                
                start_time = time.time()
                schedule_result = scheduler.schedule(
                    graph, task_list, message_list, ET, CT,
                    random_state=random_state
                )
                execution_time = time.time() - start_time
                
                # Create result
                result = ExperimentResult(
                    dag_type=config.dag_type,
                    dag_param_value=dag_param_value,
                    num_processors=num_proc,
                    num_buses=num_bus,
                    ccr=ccr,
                    algorithm=algorithm,
                    run_id=run_id,
                    makespan=schedule_result['makespan'],
                    execution_time=execution_time,
                    num_tasks=len(task_list),
                    num_messages=len(message_list),
                    converged=schedule_result.get('q_learning_converged'),
                    episodes=schedule_result.get('q_learning_episodes')
                )
                
                results.append(result)
        
        return results
    
    def _calculate_total_runs(self, config: ExperimentConfig) -> int:
        """Calculate total number of runs for progress tracking."""
        param_values = config.dag_params[list(config.dag_params.keys())[0]]
        total = (len(param_values) * len(config.processors) * 
                len(config.buses) * len(config.ccr_values) * 
                len(config.algorithms) * config.num_runs)
        return total
    
    def _generate_param_combinations(self, config: ExperimentConfig):
        """Generate all parameter combinations for the experiment."""
        param_name = list(config.dag_params.keys())[0]
        param_values = config.dag_params[param_name]
        
        combinations = []
        for param_val in param_values:
            for num_proc in config.processors:
                for num_bus in config.buses:
                    for ccr in config.ccr_values:
                        for algorithm in config.algorithms:
                            combinations.append((param_val, num_proc, num_bus, ccr, algorithm))
        
        return combinations
    
    def save_results(self, results: List[ExperimentResult], 
                    experiment_name: str) -> str:
        """
        Save experiment results to files.
        
        Args:
            results: List of experiment results
            experiment_name: Name for the experiment
            
        Returns:
            Path to saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in results])
        
        # Save as CSV
        csv_path = os.path.join(self.results_dir, f"{experiment_name}_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        # Save configuration and metadata
        metadata = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'total_runs': len(results),
            'unique_configurations': len(df.drop_duplicates(['dag_type', 'dag_param_value', 
                                                           'num_processors', 'num_buses', 
                                                           'ccr', 'algorithm'])),
            'algorithms': df['algorithm'].unique().tolist(),
            'dag_types': df['dag_type'].unique().tolist()
        }
        
        metadata_path = os.path.join(self.results_dir, f"{experiment_name}_{timestamp}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return csv_path
    
    def calculate_makespan_ratios(self, results: List[ExperimentResult]) -> pd.DataFrame:
        """
        Calculate makespan ratios between algorithms.
        
        Makespan Ratio = CC-TMS Makespan / QL-CC-TMS Makespan × 100
        
        Args:
            results: List of experiment results
            
        Returns:
            DataFrame with makespan ratios
        """
        df = pd.DataFrame([asdict(r) for r in results])
        
        # Group by experiment parameters (excluding algorithm and run_id)
        group_cols = ['dag_type', 'dag_param_value', 'num_processors', 'num_buses', 'ccr', 'run_id']
        
        ratios = []
        
        for group_key, group_df in df.groupby(group_cols):
            if len(group_df['algorithm'].unique()) >= 2:
                cctms_results = group_df[group_df['algorithm'] == 'cctms']
                qlcctms_results = group_df[group_df['algorithm'] == 'qlcctms']
                
                if not cctms_results.empty and not qlcctms_results.empty:
                    cctms_makespan = cctms_results.iloc[0]['makespan']
                    qlcctms_makespan = qlcctms_results.iloc[0]['makespan']
                    
                    ratio = (cctms_makespan / qlcctms_makespan) * 100 if qlcctms_makespan > 0 else 0
                    
                    ratio_result = {
                        'dag_type': group_key[0],
                        'dag_param_value': group_key[1],
                        'num_processors': group_key[2],
                        'num_buses': group_key[3],
                        'ccr': group_key[4],
                        'run_id': group_key[5],
                        'makespan_ratio': ratio,
                        'cctms_makespan': cctms_makespan,
                        'qlcctms_makespan': qlcctms_makespan
                    }
                    ratios.append(ratio_result)
        
        return pd.DataFrame(ratios)
    
    def generate_summary_statistics(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Generate summary statistics for experiment results."""
        df = pd.DataFrame([asdict(r) for r in results])
        
        summary = {
            'total_experiments': len(results),
            'by_algorithm': {},
            'by_dag_type': {},
            'makespan_statistics': {},
            'execution_time_statistics': {}
        }
        
        # Statistics by algorithm
        for alg in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == alg]
            summary['by_algorithm'][alg] = {
                'count': len(alg_data),
                'avg_makespan': float(alg_data['makespan'].mean()),
                'std_makespan': float(alg_data['makespan'].std()),
                'avg_execution_time': float(alg_data['execution_time'].mean()),
                'std_execution_time': float(alg_data['execution_time'].std())
            }
        
        # Statistics by DAG type
        for dag_type in df['dag_type'].unique():
            dag_data = df[df['dag_type'] == dag_type]
            summary['by_dag_type'][dag_type] = {
                'count': len(dag_data),
                'avg_makespan': float(dag_data['makespan'].mean()),
                'avg_num_tasks': float(dag_data['num_tasks'].mean()),
                'avg_num_messages': float(dag_data['num_messages'].mean())
            }
        
        # Overall makespan statistics
        summary['makespan_statistics'] = {
            'min': float(df['makespan'].min()),
            'max': float(df['makespan'].max()),
            'mean': float(df['makespan'].mean()),
            'std': float(df['makespan'].std())
        }
        
        # Overall execution time statistics
        summary['execution_time_statistics'] = {
            'min': float(df['execution_time'].min()),
            'max': float(df['execution_time'].max()),
            'mean': float(df['execution_time'].mean()),
            'std': float(df['execution_time'].std())
        }
        
        return summary