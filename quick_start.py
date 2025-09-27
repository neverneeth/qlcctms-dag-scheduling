"""
Quick Start Guide for DAG Scheduling Framework

This script demonstrates the key features of the framework.
Run this to get started quickly!
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def demonstrate_framework():
    """Demonstrate key framework capabilities."""
    print("=" * 70)
    print("DAG SCHEDULING FRAMEWORK - QUICK START DEMONSTRATION")
    print("=" * 70)
    
    try:
        # Import framework modules
        from src.dag_generators import DAGFactory
        from src.schedulers import SchedulerFactory
        from src.cost_matrices import generate_cost_matrices, validate_cost_matrices
        from src.experiment_runner import ExperimentRunner, ExperimentConfig
        
        print("✓ Framework modules imported successfully")
        
        # 1. Demonstrate DAG Generation
        print("\n1. DAG GENERATION DEMONSTRATION")
        print("-" * 40)
        
        # Test each DAG type
        dag_types = ['gaussian', 'epigenomics', 'laplace', 'stencil']
        dag_params = {'gaussian': ('chi', 4), 'epigenomics': ('gamma', 3), 
                     'laplace': ('phi', 3), 'stencil': ('xi', 3)}
        
        for dag_type in dag_types:
            generator = DAGFactory.create_generator(dag_type)
            param_name, param_value = dag_params[dag_type]
            
            if dag_type == 'gaussian':
                graph, tasks, messages = generator.generate(chi=param_value)
                expected_tasks, expected_messages = generator.get_theoretical_counts(chi=param_value)
            elif dag_type == 'epigenomics':
                graph, tasks, messages = generator.generate(gamma=param_value)
                expected_tasks, expected_messages = generator.get_theoretical_counts(gamma=param_value)
            elif dag_type == 'laplace':
                graph, tasks, messages = generator.generate(phi=param_value)
                expected_tasks, expected_messages = generator.get_theoretical_counts(phi=param_value)
            elif dag_type == 'stencil':
                graph, tasks, messages = generator.generate(xi=param_value)
                expected_tasks, expected_messages = generator.get_theoretical_counts(xi=param_value)
            
            actual_tasks, actual_messages = len(tasks), len(messages)
            validation = "✓" if (actual_tasks == expected_tasks and actual_messages == expected_messages) else "✗"
            
            print(f"  {dag_type.upper()}: {param_name}={param_value} → "
                  f"{actual_tasks}/{expected_tasks} tasks, {actual_messages}/{expected_messages} messages {validation}")
        
        # 2. Demonstrate Cost Matrix Generation
        print("\n2. COST MATRIX GENERATION")
        print("-" * 40)
        
        # Use Gaussian elimination DAG for demonstration
        generator = DAGFactory.create_generator('gaussian')
        graph, tasks, messages = generator.generate(chi=4)
        
        ET, CT, task_list, message_list = generate_cost_matrices(
            graph, num_processors=2, num_buses=2, ccr=1.0, random_state=42
        )
        
        stats = validate_cost_matrices(ET, CT, task_list, message_list)
        print(f"  Generated matrices for {stats['num_tasks']} tasks, {stats['num_messages']} messages")
        print(f"  Execution times: {stats['et_min']:.1f}-{stats['et_max']:.1f} ms (avg: {stats['et_mean']:.1f})")
        print(f"  Communication times: {stats['ct_min']:.1f}-{stats['ct_max']:.1f} ms (avg: {stats['ct_mean']:.1f})")
        print(f"  Actual CCR: {stats['actual_ccr']:.2f}")
        
        # 3. Demonstrate Scheduling Algorithms
        print("\n3. SCHEDULING ALGORITHMS DEMONSTRATION")
        print("-" * 40)
        
        algorithms = ['cctms', 'qlcctms']
        results = {}
        
        for alg in algorithms:
            scheduler = SchedulerFactory.create_scheduler(alg)
            result = scheduler.schedule(graph, tasks, messages, ET, CT, random_state=42)
            results[alg] = result
            
            print(f"  {alg.upper()}: Makespan = {result['makespan']:.2f} ms")
            if 'q_learning_episodes' in result:
                print(f"    Q-Learning episodes: {result['q_learning_episodes']}")
                print(f"    Converged: {result['q_learning_converged']}")
        
        # Calculate makespan ratio
        if len(results) == 2:
            ratio = (results['cctms']['makespan'] / results['qlcctms']['makespan']) * 100
            print(f"  Makespan Ratio (CC-TMS/QL-CC-TMS): {ratio:.2f}%")
        
        # 4. Demonstrate Experiment Configuration
        print("\n4. EXPERIMENT CONFIGURATION DEMONSTRATION")
        print("-" * 40)
        
        config = ExperimentConfig(
            dag_type='gaussian',
            dag_params={'chi': [3, 4]},
            processors=[2, 4],
            buses=[1, 2],
            ccr_values=[1.0],
            algorithms=['cctms', 'qlcctms'],
            num_runs=3,  # Small number for demo
            random_state_base=42
        )
        
        print(f"  Configuration created for {config.dag_type} DAG")
        print(f"  Parameters: {config.dag_params}")
        print(f"  Platform: {len(config.processors)} processor configs, {len(config.buses)} bus configs")
        print(f"  Algorithms: {config.algorithms}")
        print(f"  Total experiment combinations: {len(config.dag_params['chi']) * len(config.processors) * len(config.buses) * len(config.ccr_values) * len(config.algorithms) * config.num_runs}")
        
        # 5. Run Mini Experiment
        print("\n5. MINI EXPERIMENT DEMONSTRATION")
        print("-" * 40)
        
        runner = ExperimentRunner(results_dir="results")
        
        print("  Running mini experiment...")
        mini_results = runner.run_experiment(config)
        
        print(f"  Completed {len(mini_results)} experiment runs")
        
        # Save results
        results_file = runner.save_results(mini_results, "demo_experiment")
        print(f"  Results saved to: {os.path.basename(results_file)}")
        
        # Generate summary
        summary = runner.generate_summary_statistics(mini_results)
        print(f"  Summary statistics:")
        for alg, stats in summary['by_algorithm'].items():
            print(f"    {alg.upper()}: Avg makespan = {stats['avg_makespan']:.2f} ms")
        
        # Calculate ratios
        ratios_df = runner.calculate_makespan_ratios(mini_results)
        if not ratios_df.empty:
            avg_ratio = ratios_df['makespan_ratio'].mean()
            print(f"  Average makespan ratio: {avg_ratio:.2f}%")
        
        print("\n" + "=" * 70)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Run 'python experiments/demo_experiment.py' for detailed examples")
        print("2. Run 'python experiments/full_experiments.py --mode quick' for validation")
        print("3. Check the results/ directory for generated data")
        print("4. See README.md for complete documentation")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nPlease install required dependencies:")
        print("pip install networkx numpy matplotlib pandas seaborn")
        return False
        
    except Exception as e:
        print(f"✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = demonstrate_framework()
    if success:
        print("\n🎉 Framework is ready to use!")
    else:
        print("\n❌ Please resolve the issues above before using the framework.")
        sys.exit(1)