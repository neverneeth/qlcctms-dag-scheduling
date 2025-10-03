"""
Constants and Configuration Settings for DAG Scheduling Framework

This module provides centralized configuration for consistent behavior across
all components of the DAG scheduling framework.
"""

from typing import Dict, Any

# DAG Type Constants
class DAGTypes:
    """Standard DAG type names for consistent usage across the framework."""
    GAUSSIAN = 'gaussian'
    EPIGENOMICS = 'epigenomics'
    LAPLACE = 'laplace'
    STENCIL = 'stencil'
    
    @classmethod
    def all_types(cls) -> list:
        """Get all available DAG types."""
        return [cls.GAUSSIAN, cls.EPIGENOMICS, cls.LAPLACE, cls.STENCIL]
    
    @classmethod
    def validate_type(cls, dag_type: str) -> bool:
        """Validate if the given DAG type is supported."""
        return dag_type in cls.all_types()


# Algorithm Constants
class Algorithms:
    """Standard algorithm names."""
    CCTMS = 'cctms'
    QLCCTMS = 'qlcctms'
    
    @classmethod
    def all_algorithms(cls) -> list:
        """Get all available algorithms."""
        return [cls.CCTMS, cls.QLCCTMS]


# Q-Learning Default Parameters
"""DEFAULT_QL_PARAMS = {
    'epsilon': 0.2,                    # Exploration rate
    'learning_rate': 0.1,              # Learning rate (alpha)
    'discount': 0.8,                   # Discount factor (gamma)
    'max_episodes': 2000,              # Maximum episodes (reduced from 300,000)
    'convergence_window': 20,          # Window for convergence check (reduced from 40)
    'convergence_threshold': 0.5       # Convergence threshold (reduced from 0.2)
}"""
DEFAULT_QL_PARAMS = {
  'epsilon': 0.05,
  'learning_rate': 0.05,
  'discount': 0.7,
  'max_episodes': 8054,
  'convergence_window': 10,
  'convergence_threshold': 0.2
}

# Experiment Configuration
class ExperimentConfig:
    """Configuration for experiments."""
    
    # Full experiment parameters
    FULL_DAG_CONFIGS = {
        DAGTypes.GAUSSIAN: [3, 4, 5, 6],        # χ values
        DAGTypes.EPIGENOMICS: [2, 3, 4, 5],     # γ values  
        DAGTypes.LAPLACE: [2, 3, 4, 5],         # φ values
        DAGTypes.STENCIL: [2, 3, 4, 5]          # ξ values
    }
    
    # Small test parameters (for quick testing)
    SMALL_DAG_CONFIGS = {
        DAGTypes.GAUSSIAN: [5]                  # Single χ value for testing
    }
    
    # Platform configurations
    FULL_PROCESSORS = [2, 4, 6, 8]            # p values for full experiment
    FULL_BUSES = [1, 2, 3, 4]                 # b values for full experiment
    
    SMALL_PROCESSORS = [2, 4, 6, 8, 10]       # p values for small test
    SMALL_BUSES = [1]                          # b values for small test
    
    # Experiment settings
    FULL_ITERATIONS = 100                      # Iterations for statistical significance
    SMALL_ITERATIONS = 10                      # Iterations for quick testing

    CCR = [0.5, 1.0]                                  # Communication-to-Computation Ratio


# Visualization Constants
class PlotConfig:
    """Configuration for plots and visualizations."""
    
    # Plot dimensions
    BOX_PLOT_SIZE = (12, 6)
    LINE_PLOT_SIZE = (10, 6)
    
    # Colors
    QLCCTMS_COLOR = 'red'
    CCTMS_COLOR = 'blue'
    BOX_PLOT_COLOR = 'lightcoral'
    
    # Markers
    QLCCTMS_MARKER = 'o'
    CCTMS_MARKER = 's'
    
    # Labels
    QLCCTMS_LABEL = 'QL-CC-TMS'
    CCTMS_LABEL = 'CC-TMS'
    
    # File settings
    DPI = 300
    FORMAT = 'png'


# Progress Reporting
class ProgressConfig:
    """Configuration for progress reporting."""
    
    PROGRESS_INTERVAL_SMALL = 10              # Report every N configs for small tests
    PROGRESS_INTERVAL_FULL = 100             # Report every N configs for full experiments
    
    # Time estimation
    ENABLE_ETA = True
    ETA_WINDOW_SIZE = 50                     # Number of recent measurements for ETA


# File and Directory Constants
class FileConfig:
    """File and directory configuration."""
    
    DEFAULT_RESULTS_DIR = "./results"
    PLOTS_SUBDIR = "plots"
    
    # File naming patterns
    CSV_PATTERN = "{experiment_name}_{timestamp}.csv"
    METADATA_PATTERN = "{experiment_name}_{timestamp}_metadata.json"
    SUMMARY_PATTERN = "{experiment_name}_summary_statistics_{timestamp}.json"
    
    # Plot naming patterns
    BOX_PLOT_PATTERN = "boxplot_{dag_type}_param_{param}_{constraint}_{algorithm}.png"
    LINE_PLOT_PATTERN = "lineplot_{dag_type}_param_{param}_{constraint}_avg_makespan.png"


# Error Handling Configuration
class ErrorConfig:
    """Configuration for error handling and logging."""
    
    MAX_RETRIES = 3                          # Maximum retries for failed operations
    RETRY_DELAY = 1.0                        # Delay between retries (seconds)
    
    # Error tolerance
    ALLOW_PARTIAL_FAILURES = True           # Continue experiment if some configs fail
    MIN_SUCCESS_RATE = 0.8                  # Minimum success rate to consider experiment valid


# Validation Constants
class ValidationConfig:
    """Configuration for input validation."""
    
    MIN_TASKS = 1                           # Minimum number of tasks in DAG
    MAX_TASKS = 1000                        # Maximum number of tasks in DAG
    
    MIN_PROCESSORS = 1                      # Minimum number of processors
    MAX_PROCESSORS = 64                     # Maximum number of processors
    
    MIN_BUSES = 1                           # Minimum number of buses
    MAX_BUSES = 32                          # Maximum number of buses
    
    MIN_ITERATIONS = 1                      # Minimum iterations per config
    MAX_ITERATIONS = 10000                  # Maximum iterations per config