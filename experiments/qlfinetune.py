"""
Q-Learning Parameter Optimization Experiment

This experiment efficiently finds optimal Q-Learning parameters for the QL-CC-TMS algorithm
using Bayesian optimization instead of exhaustive grid search (50 trials vs 729 combinations).

Optimization Objectives:
1. Minimize makespan ratio (QL-CC-TMS/CC-TMS)
2. Minimize variance across different DAG types and system configurations
3. Maximize convergence speed (minimize episodes to convergence)

Parameter Space:
- epsilon: [0.05, 0.3] (exploration rate)
- learning_rate: [0.05, 0.3] (learning rate alpha)
- discount: [0.7, 0.9] (discount factor gamma)  
- max_episodes: [1000, 10000] (maximum training episodes)
- convergence_window: [10, 50] (convergence check window)
- convergence_threshold: [0.01, 0.2] (convergence threshold)

Strategy: Sequential Model-Based Optimization (SMBO) using Gaussian Process
- Tests ~50 parameter combinations intelligently
- Uses Bayesian optimization to focus on promising regions
- Evaluates each parameter set across diverse DAG types and platforms
- Provides statistical confidence in results with uncertainty quantification
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


class QLParameterOptimizer:
    """
    Efficient Q-Learning parameter optimization using Bayesian optimization
    or Latin Hypercube Sampling for the QL-CC-TMS algorithm.
    """
    
    def __init__(self, results_dir="./results/ql_optimization", iterations=5, n_trials=50):
        """
        Initialize the Q-Learning parameter optimizer.
        
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
            'epsilon': [0.05, 0.3],
            'learning_rate': [0.05, 0.3],
            'discount': [0.7, 0.9],
            'max_episodes': [1000, 10000],
            'convergence_window': [10, 50],
            'convergence_threshold': [0.01, 0.2]
        }
        
        # Diverse test configurations
        self.test_configs = self._create_test_configurations()
        
        # Results storage
        self.optimization_results = []
        self.best_params = None
        self.best_score = float('inf')
        
        print(f"QL Parameter Optimizer Initialized")
        print(f"Results directory: {self.results_dir}")
        print(f"Test configurations: {len(self.test_configs)}")
        print(f"Iterations per evaluation: {self.iterations}")
        print(f"Total trials: {self.n_trials}")
        
    def _create_test_configurations(self):
        """Create diverse test configurations for parameter evaluation."""
        configs = []
        
        # DAG configurations (representative samples from each type)
        dag_configs = {
            DAGTypes.GAUSSIAN: [3, 5, 6],      # Small, medium, large
            DAGTypes.EPIGENOMICS: [2, 3, 4],   # Small, medium, large  
            DAGTypes.LAPLACE: [2, 3, 4],       # Small, medium, large
            DAGTypes.STENCIL: [2, 3, 4]        # Small, medium, large
        }
        
        # Platform configurations (diverse processor/bus combinations)
        platform_configs = [
            (2, 1), (4, 1), (6, 2), (8, 2)    # Different scales
        ]
        
        # Create test configurations
        for dag_type, params in dag_configs.items():
            for param in params:
                for processors, buses in platform_configs:
                    configs.append({
                        'dag_type': dag_type,
                        'dag_param': param,
                        'processors': processors,
                        'buses': buses
                    })
        
        return configs
    
    def evaluate_parameter_set(self, params):
        """
        Evaluate a single parameter set across all test configurations.
        
        Args:
            params (dict): Q-Learning parameters to evaluate
            
        Returns:
            dict: Evaluation results including score, metrics, and statistics
        """
        start_time = time.time()
        results = []
        
        # Test across all configurations
        for config in self.test_configs:
            try:
                # Generate DAG
                dag_generator = DAGFactory.create_generator(config['dag_type'])
                
                # Generate DAG with appropriate parameter based on type
                if config['dag_type'] == DAGTypes.GAUSSIAN:
                    dag, task_list, message_list = dag_generator.generate(chi=config['dag_param'])
                elif config['dag_type'] == DAGTypes.EPIGENOMICS:
                    dag, task_list, message_list = dag_generator.generate(gamma=config['dag_param'])
                elif config['dag_type'] == DAGTypes.LAPLACE:
                    dag, task_list, message_list = dag_generator.generate(phi=config['dag_param'])
                elif config['dag_type'] == DAGTypes.STENCIL:
                    dag, task_list, message_list = dag_generator.generate(xi=config['dag_param'])
                else:
                    raise ValueError(f"Unknown DAG type: {config['dag_type']}")
                
                # Generate cost matrices
                ET, CT, TL, ML = generate_cost_matrices(
                    dag, config['processors'], config['buses'], ccr=1.0, random_state=42
                )
                
                # Test CC-TMS (baseline)
                cctms_scheduler = SchedulerUtils.create_scheduler_with_params(Algorithms.CCTMS)
                cctms_result = cctms_scheduler.schedule(dag, task_list, message_list, ET, CT, random_state=42)
                cctms_makespan = cctms_result['makespan']
                
                # Test QL-CC-TMS with current parameters
                ql_scheduler = SchedulerUtils.create_scheduler_with_params(Algorithms.QLCCTMS, params)
                
                # Run multiple iterations for statistical reliability
                ql_makespans = []
                ql_episodes = []
                ql_converged = []
                
                for iteration in range(self.iterations):
                    ql_result = ql_scheduler.schedule(
                        dag, task_list, message_list, ET, CT, 
                        random_state=42 + iteration
                    )
                    ql_makespans.append(ql_result['makespan'])
                    ql_episodes.append(ql_result.get('q_learning_episodes', 0))
                    ql_converged.append(ql_result.get('q_learning_converged', False))
                
                # Calculate metrics for this configuration
                avg_ql_makespan = np.mean(ql_makespans)
                makespan_ratio = avg_ql_makespan / cctms_makespan
                makespan_std = np.std(ql_makespans)
                avg_episodes = np.mean(ql_episodes)
                convergence_rate = np.mean(ql_converged)
                
                results.append({
                    'dag_type': config['dag_type'],
                    'dag_param': config['dag_param'],
                    'processors': config['processors'],
                    'buses': config['buses'],
                    'cctms_makespan': cctms_makespan,
                    'ql_makespan': avg_ql_makespan,
                    'makespan_ratio': makespan_ratio,
                    'makespan_std': makespan_std,
                    'episodes': avg_episodes,
                    'convergence_rate': convergence_rate
                })
                
            except Exception as e:
                print(f"Error in configuration {config}: {e}")
                continue
        
        # Calculate overall metrics
        if not results:
            return {'score': float('inf'), 'results': [], 'error': 'No valid results'}
        
        makespan_ratios = [r['makespan_ratio'] for r in results]
        episodes_list = [r['episodes'] for r in results]
        convergence_rates = [r['convergence_rate'] for r in results]
        
        # Composite score: 60% makespan ratio + 25% variance + 15% convergence efficiency
        avg_ratio = np.mean(makespan_ratios)
        ratio_variance = np.var(makespan_ratios)
        avg_episodes_normalized = np.mean(episodes_list) / params['max_episodes']  # Normalize by max
        avg_convergence = np.mean(convergence_rates)
        
        # Lower score is better
        score = (0.6 * avg_ratio + 
                0.25 * ratio_variance + 
                0.15 * avg_episodes_normalized - 
                0.1 * avg_convergence)  # Bonus for convergence
        
        evaluation_time = time.time() - start_time
        
        return {
            'score': score,
            'avg_makespan_ratio': avg_ratio,
            'ratio_variance': ratio_variance,
            'avg_episodes': np.mean(episodes_list),
            'avg_convergence_rate': avg_convergence,
            'evaluation_time': evaluation_time,
            'results': results,
            'params': params.copy()
        }
    
    def latin_hypercube_sampling(self, n_samples):
        """Generate parameter combinations using Latin Hypercube Sampling."""
        from scipy.stats import qmc
        
        # Create sampler
        sampler = qmc.LatinHypercube(d=len(self.param_bounds), seed=42)
        samples = sampler.random(n=n_samples)
        
        # Scale samples to parameter bounds
        param_combinations = []
        param_names = list(self.param_bounds.keys())
        
        for sample in samples:
            params = {}
            for i, param_name in enumerate(param_names):
                lower, upper = self.param_bounds[param_name]
                if param_name in ['max_episodes', 'convergence_window']:
                    # Integer parameters
                    params[param_name] = int(lower + sample[i] * (upper - lower))
                else:
                    # Float parameters
                    params[param_name] = lower + sample[i] * (upper - lower)
            param_combinations.append(params)
        
        return param_combinations
    
    def bayesian_optimization(self):
        """Run Bayesian optimization using scikit-optimize."""
        if not BAYESIAN_AVAILABLE:
            print("Bayesian optimization not available. Using Latin Hypercube Sampling.")
            return self.latin_hypercube_optimization()
        
        # Define parameter space for skopt
        space = [
            Real(self.param_bounds['epsilon'][0], self.param_bounds['epsilon'][1], name='epsilon'),
            Real(self.param_bounds['learning_rate'][0], self.param_bounds['learning_rate'][1], name='learning_rate'),
            Real(self.param_bounds['discount'][0], self.param_bounds['discount'][1], name='discount'),
            Integer(self.param_bounds['max_episodes'][0], self.param_bounds['max_episodes'][1], name='max_episodes'),
            Integer(self.param_bounds['convergence_window'][0], self.param_bounds['convergence_window'][1], name='convergence_window'),
            Real(self.param_bounds['convergence_threshold'][0], self.param_bounds['convergence_threshold'][1], name='convergence_threshold')
        ]
        
        @use_named_args(space)
        def objective(**params):
            """Objective function for Bayesian optimization."""
            result = self.evaluate_parameter_set(params)
            self.optimization_results.append(result)
            
            # Update best parameters
            if result['score'] < self.best_score:
                self.best_score = result['score']
                self.best_params = params.copy()
                print(f"New best score: {self.best_score:.4f} with params: {params}")
            
            return result['score']
        
        print("Starting Bayesian optimization...")
        print(f"Parameter space: {[dim.name for dim in space]}")
        
        # Run optimization
        result = gp_minimize(objective, space, n_calls=self.n_trials, random_state=42, verbose=True)
        
        return result
    
    def latin_hypercube_optimization(self):
        """Run optimization using Latin Hypercube Sampling."""
        print("Starting Latin Hypercube Sampling optimization...")
        
        # Generate parameter combinations
        param_combinations = self.latin_hypercube_sampling(self.n_trials)
        
        # Evaluate each combination
        for i, params in enumerate(param_combinations):
            print(f"Evaluating combination {i+1}/{self.n_trials}: {params}")
            
            result = self.evaluate_parameter_set(params)
            self.optimization_results.append(result)
            
            # Update best parameters
            if result['score'] < self.best_score:
                self.best_score = result['score']
                self.best_params = params.copy()
                print(f"New best score: {self.best_score:.4f}")
        
        return self.optimization_results
    
    def optimize(self, method='auto'):
        """
        Run parameter optimization.
        
        Args:
            method (str): 'bayesian', 'latin_hypercube', or 'auto'
        """
        start_time = datetime.now()
        print("\n" + "="*80)
        print("Q-LEARNING PARAMETER OPTIMIZATION")
        print("="*80)
        
        if method == 'auto':
            method = 'bayesian' if BAYESIAN_AVAILABLE else 'latin_hypercube'
        
        print(f"Using {method} optimization")
        print(f"Testing {self.n_trials} parameter combinations")
        print(f"Across {len(self.test_configs)} test configurations")
        print(f"With {self.iterations} iterations each")
        
        # Run optimization
        if method == 'bayesian':
            optimization_result = self.bayesian_optimization()
        else:
            optimization_result = self.latin_hypercube_optimization()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print(f"\nOptimization completed in {duration:.1f} minutes")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        # Save results
        self.save_results()
        self.analyze_results()
        self.generate_visualizations()
        
        return self.best_params, self.best_score
    
    def save_results(self):
        """Save optimization results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_df = pd.DataFrame([
            {**result['params'], **{k: v for k, v in result.items() if k != 'results' and k != 'params'}}
            for result in self.optimization_results
        ])
        
        csv_path = os.path.join(self.results_dir, f"ql_optimization_results_{timestamp}.csv")
        results_df.to_csv(csv_path, index=False)
        
        # Save best parameters
        best_params_path = os.path.join(self.results_dir, f"best_ql_parameters_{timestamp}.json")
        with open(best_params_path, 'w') as f:
            json.dump({
                'best_parameters': self.best_params,
                'best_score': self.best_score,
                'optimization_timestamp': timestamp,
                'n_trials': self.n_trials,
                'test_configurations': len(self.test_configs)
            }, f, indent=2)
        
        print(f"Results saved to: {csv_path}")
        print(f"Best parameters saved to: {best_params_path}")
    
    def analyze_results(self):
        """Analyze optimization results and generate insights."""
        if not self.optimization_results:
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {**result['params'], **{k: v for k, v in result.items() if k not in ['results', 'params']}}
            for result in self.optimization_results
        ])
        
        print("\n" + "="*60)
        print("OPTIMIZATION ANALYSIS")
        print("="*60)
        
        # Parameter sensitivity analysis
        param_correlations = {}
        for param in self.param_bounds.keys():
            correlation = df[param].corr(df['score'])
            param_correlations[param] = correlation
            print(f"{param}: correlation with score = {correlation:.3f}")
        
        # Best vs worst comparison
        best_idx = df['score'].idxmin()
        worst_idx = df['score'].idxmax()
        
        print(f"\nBest configuration (score: {df.loc[best_idx, 'score']:.4f}):")
        for param in self.param_bounds.keys():
            print(f"  {param}: {df.loc[best_idx, param]}")
        
        print(f"\nWorst configuration (score: {df.loc[worst_idx, 'score']:.4f}):")
        for param in self.param_bounds.keys():
            print(f"  {param}: {df.loc[worst_idx, param]}")
        
        # Performance statistics
        print(f"\nPerformance Statistics:")
        print(f"Best makespan ratio: {df['avg_makespan_ratio'].min():.4f}")
        print(f"Average makespan ratio: {df['avg_makespan_ratio'].mean():.4f}")
        print(f"Lowest variance: {df['ratio_variance'].min():.6f}")
        print(f"Highest convergence rate: {df['avg_convergence_rate'].max():.3f}")
        print(f"Average episodes (best): {df.loc[best_idx, 'avg_episodes']:.0f}")
    
    def generate_visualizations(self):
        """Generate optimization result visualizations."""
        if not self.optimization_results:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {**result['params'], **{k: v for k, v in result.items() if k not in ['results', 'params']}}
            for result in self.optimization_results
        ])
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Optimization progress plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Q-Learning Parameter Optimization Results', fontsize=16)
        
        # Score over trials
        axes[0, 0].plot(range(1, len(df) + 1), df['score'], 'b-', alpha=0.7, linewidth=2)
        axes[0, 0].plot(range(1, len(df) + 1), df['score'].cummin(), 'r-', linewidth=2, label='Best so far')
        axes[0, 0].set_xlabel('Trial')
        axes[0, 0].set_ylabel('Score (lower is better)')
        axes[0, 0].set_title('Optimization Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Parameter correlation heatmap
        param_cols = list(self.param_bounds.keys())
        corr_matrix = df[param_cols + ['score']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 1])
        axes[0, 1].set_title('Parameter Correlations')
        
        # Makespan ratio distribution
        axes[1, 0].hist(df['avg_makespan_ratio'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(df['avg_makespan_ratio'].min(), color='red', linestyle='--', 
                          label=f'Best: {df["avg_makespan_ratio"].min():.3f}')
        axes[1, 0].set_xlabel('Average Makespan Ratio')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Makespan Ratio Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Episodes vs Convergence Rate
        scatter = axes[1, 1].scatter(df['avg_episodes'], df['avg_convergence_rate'], 
                                   c=df['score'], cmap='viridis', alpha=0.7, s=50)
        axes[1, 1].set_xlabel('Average Episodes')
        axes[1, 1].set_ylabel('Convergence Rate')
        axes[1, 1].set_title('Episodes vs Convergence Rate')
        plt.colorbar(scatter, ax=axes[1, 1], label='Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'plots', 'optimization_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Parameter sensitivity plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Parameter Sensitivity Analysis', fontsize=16)
        
        param_names = list(self.param_bounds.keys())
        for i, param in enumerate(param_names):
            row, col = i // 3, i % 3
            axes[row, col].scatter(df[param], df['score'], alpha=0.6, s=40)
            axes[row, col].set_xlabel(param)
            axes[row, col].set_ylabel('Score')
            axes[row, col].set_title(f'{param} vs Score')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr = df[param].corr(df['score'])
            axes[row, col].text(0.05, 0.95, f'r = {corr:.3f}', 
                              transform=axes[row, col].transAxes, 
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        sensitivity_path = os.path.join(self.results_dir, 'plots', 'parameter_sensitivity.png')
        plt.savefig(sensitivity_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to: {os.path.join(self.results_dir, 'plots')}")


def main():
    """Main function to run Q-Learning parameter optimization."""
    print("Starting Q-Learning Parameter Optimization...")
    
    # Create optimizer
    optimizer = QLParameterOptimizer(
        results_dir="./results/ql_optimization",
        iterations=3,  # Reduced for faster testing, increase for production
        n_trials=25    # Reduced for testing, increase to 50+ for production
    )
    
    # Run optimization
    best_params, best_score = optimizer.optimize(method='auto')
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    print(f"Best Parameters Found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"\nBest Score: {best_score:.4f}")
    print(f"\nRecommendation: Update DEFAULT_QL_PARAMS in config/constants.py with these values")
    print("="*80)


if __name__ == "__main__":
    main()