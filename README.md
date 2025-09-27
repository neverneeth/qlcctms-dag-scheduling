# DAG Scheduling Framework

A comprehensive, modular framework for experimenting with DAG (Directed Acyclic Graph) scheduling algorithms in distributed computing systems.

## Overview

This framework implements the experimental setup described in the research paper for evaluating Communication-Conscious Task and Message Scheduling (CC-TMS) and Q-Learning based CC-TMS (QL-CC-TMS) algorithms.

## Framework Architecture

The framework is organized into the following modules:

```
dag_scheduling_framework/
├── src/                          # Core framework modules
│   ├── dag_generators.py         # DAG generation for benchmark problems
│   ├── schedulers.py             # Scheduling algorithms implementation
│   ├── cost_matrices.py          # Cost matrix generation
│   ├── experiment_runner.py      # Experiment orchestration
│   └── visualization.py          # Plotting and visualization
├── config/                       # Configuration files
│   └── experiment_configs.py     # Predefined experiment configurations
├── experiments/                  # Experiment scripts
│   ├── demo_experiment.py        # Basic demonstration
│   ├── full_experiments.py       # Comprehensive experiments
│   └── analyze_results.py        # Results analysis
└── results/                      # Generated results and plots
```

## Major Changes from Original Code

### 1. **Modular Architecture**
- **Before**: Single monolithic file with all functionality
- **After**: Separated into focused modules with clear responsibilities
- **Benefit**: Easier maintenance, testing, and extension

### 2. **Object-Oriented Design**
- **Before**: Procedural functions with global variables
- **After**: Abstract base classes and factory patterns
- **Benefit**: Extensible design for adding new DAG types and algorithms

### 3. **Configuration Management**
- **Before**: Hardcoded parameters throughout the code
- **After**: Centralized configuration system with predefined experiment setups
- **Benefit**: Easy parameter adjustment and experiment replication

### 4. **Experiment Orchestration**
- **Before**: Manual execution with individual function calls
- **After**: Automated experiment runner with progress tracking and result management
- **Benefit**: Scalable experiment execution with thousands of runs

### 5. **Results Management**
- **Before**: Print statements with no persistent storage
- **After**: Structured data collection with CSV export and statistical analysis
- **Benefit**: Reproducible results and comprehensive analysis capabilities

### 6. **Visualization System**
- **Before**: Basic matplotlib plots mixed with algorithm code
- **After**: Dedicated visualization module with multiple plot types
- **Benefit**: Professional publication-ready visualizations

### 7. **Reproducibility Enhancements**
- **Before**: Partial random state management
- **After**: Complete random state control across all components
- **Benefit**: Reproducible experiments across different runs

### 8. **Performance Tracking**
- **Before**: No execution time measurement
- **After**: Detailed performance metrics including algorithm execution time
- **Benefit**: Algorithm efficiency comparison

## Benchmark DAG Implementations

The framework implements four benchmark DAG types exactly as specified in the research paper:

### 1. Gaussian Elimination DAG
- **Parameter**: χ (number of linear equations)
- **Task nodes**: (χ² + χ - 2) / 2
- **Message nodes**: χ² - χ - 1
- **Test values**: χ = {3, 4, 5, 6}

### 2. Epigenomics DAG
- **Parameter**: γ (parallel branches)
- **Task nodes**: 4γ + 4
- **Message nodes**: 5γ + 2
- **Test values**: γ = {2, 3, 4, 5}

### 3. Laplace DAG
- **Parameter**: ϕ (matrix size)
- **Task nodes**: ϕ²
- **Message nodes**: 2ϕ² - 2ϕ
- **Test values**: ϕ = {2, 3, 4, 5}

### 4. Stencil DAG
- **Parameter**: ξ (levels and tasks per level, λ = ξ)
- **Task nodes**: λ × ξ = ξ²
- **Message nodes**: (λ - 1) × (3ξ - 2) = (ξ - 1) × (3ξ - 2)
- **Test values**: ξ = {2, 3, 4, 5}

## Scheduling Algorithms

### 1. CC-TMS (Communication-Conscious Task and Message Scheduling)
- List scheduling with upward rank priority
- Considers both task execution and message communication costs
- Optimizes processor and bus assignments simultaneously

### 2. QL-CC-TMS (Q-Learning based CC-TMS)
- Reinforcement learning approach to task prioritization
- Uses ε-greedy exploration with configurable parameters
- Learns optimal task ordering through experience

## Experimental Parameters

The framework supports the complete experimental setup from the research paper:

- **Processors**: p = {2, 4, 6, 8}
- **Buses**: b = {1, 2, 3, 4}
- **Communication-to-Computation Ratio**: CCR = {0.5, 1.0, 1.5, 2.0}
- **Execution Time Distribution**: Uniform [10ms, 30ms]
- **Communication Time Distribution**: Uniform [10ms, 30ms] (scaled by CCR)
- **Number of Runs**: 100 for CC-TMS, 50 for computationally intensive algorithms

## Performance Metrics

### Primary Metric: Makespan Ratio
```
Makespan Ratio = (CC-TMS Makespan / QL-CC-TMS Makespan) × 100
```

### Additional Metrics
- Individual algorithm makespans
- Algorithm execution times
- Q-learning convergence statistics
- Statistical analysis (mean, std dev, min, max)

## Usage Examples

### Basic Usage
```python
from src.dag_generators import DAGFactory
from src.schedulers import SchedulerFactory
from src.cost_matrices import generate_cost_matrices

# Generate a DAG
generator = DAGFactory.create_generator('gaussian')
graph, tasks, messages = generator.generate(chi=4)

# Generate cost matrices
ET, CT, _, _ = generate_cost_matrices(graph, 2, 2, ccr=1.0)

# Run scheduling
scheduler = SchedulerFactory.create_scheduler('cctms')
result = scheduler.schedule(graph, tasks, messages, ET, CT)
```

### Running Experiments
```python
from src.experiment_runner import ExperimentRunner, ExperimentConfig

# Create experiment configuration
config = ExperimentConfig(
    dag_type='gaussian',
    dag_params={'chi': [3, 4, 5, 6]},
    processors=[2, 4, 6, 8],
    buses=[1, 2, 3, 4],
    ccr_values=[0.5, 1.0, 1.5, 2.0],
    algorithms=['cctms', 'qlcctms'],
    num_runs=100
)

# Run experiments
runner = ExperimentRunner()
results = runner.run_experiment(config)
runner.save_results(results, "my_experiment")
```

### Visualization
```python
from src.visualization import DAGVisualizer, ScheduleVisualizer, ResultsVisualizer

# Visualize DAG structure
DAGVisualizer.visualize_dag(graph, "My DAG")

# Visualize schedule
ScheduleVisualizer.plot_gantt_chart(result, tasks, messages, "My Schedule")

# Analyze results
ResultsVisualizer.plot_makespan_comparison(results_df)
ResultsVisualizer.plot_makespan_ratios(ratios_df)
```

## Installation and Dependencies

### Required Dependencies
```bash
pip install networkx numpy matplotlib pandas seaborn
```

### Optional Dependencies for Enhanced Features
```bash
pip install jupyter scipy scikit-learn
```

## Running Experiments

### Quick Validation
```bash
cd experiments
python demo_experiment.py
```

### Full Experimental Suite
```bash
cd experiments
python full_experiments.py --mode quick    # Quick validation
python full_experiments.py --mode full     # Full experiments (WARNING: Very long runtime)
```

### Results Analysis
```bash
cd experiments
python analyze_results.py results/experiment_results.csv --plots --report
```

## Scalability and Performance

### Computational Complexity
- **CC-TMS**: O(v × p × b) where v is tasks, p is processors, b is buses
- **QL-CC-TMS**: O(episodes × v²) where episodes can be up to 300,000

### Scalability Features
- Progress tracking for long-running experiments
- Batch result processing
- Memory-efficient data structures
- Configurable algorithm parameters for performance tuning

### Performance Optimizations
- Vectorized numpy operations for cost matrix generation
- Efficient graph algorithms using NetworkX
- Optimized Q-learning implementation with early convergence detection

## Extensibility

### Adding New DAG Types
1. Inherit from `DAGGenerator` base class
2. Implement `generate()` and `get_theoretical_counts()` methods
3. Register with `DAGFactory`

### Adding New Scheduling Algorithms
1. Inherit from `SchedulingAlgorithm` base class
2. Implement `schedule()` method
3. Register with `SchedulerFactory`

### Adding New Visualizations
1. Add methods to appropriate visualizer class
2. Follow the established interface patterns
3. Include save functionality and proper formatting

## Validation and Testing

### DAG Generation Validation
- Theoretical node counts verified against actual generated counts
- Graph structure validation (DAG properties, connectivity)
- Parameter boundary testing

### Algorithm Validation
- Makespan calculation verification
- Resource assignment validation
- Convergence testing for Q-learning

### Reproducibility Testing
- Fixed random seed experiments
- Cross-platform result validation
- Statistical consistency checks

## Results Format

### CSV Output Columns
- `dag_type`: Type of DAG (gaussian, epigenomics, laplace, stencil)
- `dag_param_value`: Parameter value for DAG generation
- `num_processors`: Number of processors
- `num_buses`: Number of buses
- `ccr`: Communication-to-Computation Ratio
- `algorithm`: Scheduling algorithm used
- `run_id`: Experiment run identifier
- `makespan`: Resulting makespan (ms)
- `execution_time`: Algorithm execution time (seconds)
- `num_tasks`: Number of tasks in DAG
- `num_messages`: Number of messages in DAG
- `converged`: Q-learning convergence status (for QL-CC-TMS)
- `episodes`: Number of Q-learning episodes (for QL-CC-TMS)

### Statistical Analysis
- Mean and standard deviation for all metrics
- Makespan ratio calculations
- Performance comparisons by DAG type, platform configuration, and algorithm
- Confidence intervals and significance testing capabilities

This framework provides a solid foundation for conducting rigorous experimental evaluations of DAG scheduling algorithms, with built-in support for the specific experimental setup described in your research paper.