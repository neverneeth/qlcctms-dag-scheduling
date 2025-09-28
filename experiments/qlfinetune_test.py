"""
Quick Test of Q-Learning Parameter Optimization

Minimal test version with reduced configurations for validation.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple, Any, Optional
import concurrent.futures
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import framework components
from config.constants import DAGTypes, Algorithms, DEFAULT_QL_PARAMS
from src.dag_generators import DAGFactory
from src.schedulers import SchedulerFactory
from src.cost_matrices import generate_cost_matrices
from src.experiment_utils import ExperimentUtils, SchedulerUtils

# Try to import scikit-optimize for Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-optimize not available. Using Latin Hypercube Sampling instead.")
    BAYESIAN_AVAILABLE = False


class QLParameterOptimizerTest:
    """
    Quick test version with minimal configurations.
    """
    
    def __init__(self, results_dir="./results/ql_optimization_test", iterations=2, n_trials=5):
        """
        Initialize the Q-Learning parameter optimizer for testing.
        
        Args:
            results_dir (str): Directory to save optimization results
            iterations (int): Number of iterations per parameter evaluation
            n_trials (int): Number of parameter combinations to test
        """
        self.results_dir = results_dir
        self.iterations = iterations
        self.n_trials = n_trials
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'plots'), exist_ok=True)
        
        # Define parameter space
        self.param_bounds = {
            'epsilon': [0.1, 0.3],
            'learning_rate': [0.1, 0.3],
            'discount': [0.7, 0.9],
            'max_episodes': [5000, 15000],
            'convergence_window': [15, 30],
            'convergence_threshold': [0.05, 0.15]
        }
        
        # Minimal test configurations (just 4 configs)
        self.test_configs = self._create_test_configurations()
        
        # Results storage
        self.optimization_results = []
        self.best_params = None
        self.best_score = float('inf')
        
        print(f"QL Parameter Optimizer Test Initialized")
        print(f"Results directory: {self.results_dir}")
        print(f"Test configurations: {len(self.test_configs)}")
        print(f"Iterations per evaluation: {self.iterations}")
        print(f"Total trials: {self.n_trials}")
        
    def _create_test_configurations(self):
        """Create minimal test configurations for parameter evaluation."""
        configs = []
        
        # Minimal DAG configurations (just one from each type)
        dag_configs = {
            DAGTypes.GAUSSIAN: [3],      # Small only
            DAGTypes.EPIGENOMICS: [2],   # Small only
        }
        
        # Minimal platform configurations
        processors = [4]     # Just 4 processors
        buses = [1]          # Just 1 bus
        
        # Generate configurations
        for dag_type in dag_configs:
            for param in dag_configs[dag_type]:
                for p in processors:
                    for b in buses:
                        configs.append({
                            'dag_type': dag_type,
                            'dag_param': param,
                            'processors': p,
                            'buses': b
                        })
        
        return configs
    
    def evaluate_parameter_set(self, params: Dict[str, float]) -> float:
        """
        Evaluate a parameter set across all test configurations.
        
        Args:
            params (Dict[str, float]): Q-Learning parameters to evaluate
            
        Returns:
            float: Composite score (lower is better)
        """
        makespan_ratios = []
        convergence_episodes = []
        
        print(f"  Evaluating params: epsilon={params['epsilon']:.3f}, lr={params['learning_rate']:.3f}, "
              f"discount={params['discount']:.3f}, max_episodes={int(params['max_episodes'])}, "
              f"conv_window={int(params['convergence_window'])}, conv_threshold={params['convergence_threshold']:.3f}")
        
        for config in self.test_configs:
            try:
                # Generate DAG
                dag_generator = DAGFactory.create_dag_generator(config['dag_type'])
                
                if config['dag_type'] == DAGTypes.GAUSSIAN:
                    dag, task_list, message_list = dag_generator.generate(chi=config['dag_param'])
                elif config['dag_type'] == DAGTypes.EPIGENOMICS:
                    dag, task_list, message_list = dag_generator.generate(gamma=config['dag_param'])
                else:
                    print(f"    Error: Unsupported DAG type {config['dag_type']}")
                    continue
                
                # Generate cost matrices
                ET, CT = generate_cost_matrices(
                    len(task_list), 
                    config['processors'], 
                    config['buses'], 
                    len(message_list)
                )
                
                # Create schedulers
                cctms_scheduler = SchedulerUtils.create_scheduler_with_params(Algorithms.CCTMS)
                ql_scheduler = SchedulerUtils.create_scheduler_with_params(Algorithms.QLCCTMS, params)
                
                # Run multiple iterations for this configuration
                config_ratios = []
                config_episodes = []
                
                for iteration in range(self.iterations):
                    # CC-TMS scheduling
                    cctms_result = cctms_scheduler.schedule(dag, task_list, message_list, ET, CT)
                    
                    # QL-CC-TMS scheduling
                    ql_result = ql_scheduler.schedule(dag, task_list, message_list, ET, CT)
                    
                    if cctms_result and ql_result and cctms_result['makespan'] > 0:
                        ratio = ql_result['makespan'] / cctms_result['makespan']
                        config_ratios.append(ratio)
                        
                        if 'episodes' in ql_result:
                            config_episodes.append(ql_result['episodes'])
                
                if config_ratios:
                    makespan_ratios.extend(config_ratios)
                    if config_episodes:
                        convergence_episodes.extend(config_episodes)
                
            except Exception as e:
                print(f"    Error in configuration {config}: {e}")
                continue
        
        if not makespan_ratios:
            print("    No valid results - returning high penalty score")
            return 10.0  # High penalty for failed evaluations
        
        # Calculate composite score
        avg_ratio = np.mean(makespan_ratios)
        ratio_variance = np.var(makespan_ratios)
        avg_episodes = np.mean(convergence_episodes) if convergence_episodes else 25000
        
        # Composite score: 60% makespan ratio + 25% variance + 15% convergence efficiency
        normalized_episodes = min(avg_episodes / 25000, 1.0)  # Normalize to [0,1]
        score = 0.6 * avg_ratio + 0.25 * ratio_variance + 0.15 * normalized_episodes
        
        print(f"    Score: {score:.4f} (ratio={avg_ratio:.3f}, var={ratio_variance:.4f}, episodes={avg_episodes:.0f})")
        
        return score
    
    def bayesian_optimization(self):
        """Run Bayesian optimization using scikit-optimize."""
        # Define the space
        space = [
            Real(self.param_bounds['epsilon'][0], self.param_bounds['epsilon'][1], name='epsilon'),
            Real(self.param_bounds['learning_rate'][0], self.param_bounds['learning_rate'][1], name='learning_rate'),
            Real(self.param_bounds['discount'][0], self.param_bounds['discount'][1], name='discount'),
            Integer(int(self.param_bounds['max_episodes'][0]), int(self.param_bounds['max_episodes'][1]), name='max_episodes'),
            Integer(int(self.param_bounds['convergence_window'][0]), int(self.param_bounds['convergence_window'][1]), name='convergence_window'),
            Real(self.param_bounds['convergence_threshold'][0], self.param_bounds['convergence_threshold'][1], name='convergence_threshold')
        ]
        
        @use_named_args(space)
        def objective(**params):
            """Objective function for Bayesian optimization."""
            result = self.evaluate_parameter_set(params)
            
            # Store result
            self.optimization_results.append({
                'params': params.copy(),
                'score': result,
                'timestamp': datetime.now()
            })
            
            return result
        
        print(f"Starting Bayesian optimization...")
        print(f"Parameter space: {[dim.name for dim in space]}")
        
        result = gp_minimize(objective, space, n_calls=self.n_trials, random_state=42, verbose=True)
        
        return result
    
    def optimize(self, method='auto'):
        """
        Run parameter optimization.
        
        Args:
            method (str): Optimization method ('bayesian', 'latin_hypercube', or 'auto')
            
        Returns:
            Tuple[Dict[str, float], float]: Best parameters and best score
        """
        print("="*80)
        print("Q-LEARNING PARAMETER OPTIMIZATION TEST")
        print("="*80)
        
        if method == 'auto':
            method = 'bayesian' if BAYESIAN_AVAILABLE else 'latin_hypercube'
        
        print(f"Using {method} optimization")
        print(f"Testing {self.n_trials} parameter combinations")
        print(f"Across {len(self.test_configs)} test configurations")
        print(f"With {self.iterations} iterations each")
        
        start_time = time.time()
        
        if method == 'bayesian' and BAYESIAN_AVAILABLE:
            optimization_result = self.bayesian_optimization()
            
            # Extract best parameters
            best_params = {
                'epsilon': optimization_result.x[0],
                'learning_rate': optimization_result.x[1],
                'discount': optimization_result.x[2],
                'max_episodes': int(optimization_result.x[3]),
                'convergence_window': int(optimization_result.x[4]),
                'convergence_threshold': optimization_result.x[5]
            }
            best_score = optimization_result.fun
        else:
            print("Fallback method not implemented in test version")
            return None, None
        
        end_time = time.time()
        optimization_time = end_time - start_time
        
        print(f"\nOptimization completed in {optimization_time:.2f} seconds")
        print(f"Best score: {best_score:.6f}")
        print("Best parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        self.best_params = best_params
        self.best_score = best_score
        
        return best_params, best_score


def main():
    """Main function to run the Q-Learning parameter optimization test."""
    print("Starting Q-Learning Parameter Optimization Test...")
    
    # Create optimizer with minimal settings
    optimizer = QLParameterOptimizerTest(
        results_dir="./results/ql_optimization_test",
        iterations=2,  # Only 2 iterations per test
        n_trials=5     # Only 5 trials total
    )
    
    # Run optimization
    best_params, best_score = optimizer.optimize(method='auto')
    
    if best_params:
        print(f"\n" + "="*80)
        print("OPTIMIZATION TEST COMPLETED")
        print("="*80)
        print(f"Best Score: {best_score:.6f}")
        print("Optimal Parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        # Save results
        results = {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_results': [
                {
                    'params': result['params'],
                    'score': result['score'],
                    'timestamp': result['timestamp'].isoformat()
                }
                for result in optimizer.optimization_results
            ]
        }
        
        results_file = os.path.join(optimizer.results_dir, 'optimization_test_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    else:
        print("Optimization failed!")


if __name__ == "__main__":
    main()