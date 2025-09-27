# Experiment Configuration for DAG Scheduling Framework
#
# This file contains predefined experiment configurations based on
# the experimental setup described in the research paper.

# DAG Parameters based on research paper specifications
DAG_CONFIGS = {
    "gaussian_elimination": {
        "dag_type": "gaussian",
        "dag_params": {"chi": [3, 4, 5, 6]},
        "description": "Gaussian Elimination with χ linear equations"
    },
    "epigenomics": {
        "dag_type": "epigenomics", 
        "dag_params": {"gamma": [2, 3, 4, 5]},
        "description": "Epigenomics with γ parallel branches"
    },
    "laplace": {
        "dag_type": "laplace",
        "dag_params": {"phi": [2, 3, 4, 5]},
        "description": "Laplace with ϕ×ϕ matrix"
    },
    "stencil": {
        "dag_type": "stencil",
        "dag_params": {"xi": [2, 3, 4, 5]},
        "description": "Stencil with ξ levels and ξ tasks per level"
    }
}

# Platform configurations
PLATFORM_CONFIGS = {
    "processors": [2, 4, 6, 8],
    "buses": [1, 2, 3, 4]
}

# Communication-to-Computation Ratios
CCR_VALUES = [0.5, 1.0, 1.5, 2.0]

# Algorithm configurations
ALGORITHM_CONFIGS = {
    "algorithms": ["cctms", "qlcctms"],
    "qlcctms_params": {
        "epsilon": 0.2,
        "learning_rate": 0.1,
        "discount": 0.8,
        "max_episodes": 300000,
        "convergence_window": 40,
        "convergence_threshold": 0.2
    }
}

# Execution and Communication Time ranges (ms)
TIME_CONFIGS = {
    "et_min": 10.0,
    "et_max": 30.0,
    "ct_min": 10.0,
    "ct_max": 30.0
}

# Experiment run configurations
RUN_CONFIGS = {
    "num_runs_full": 100,    # For CC-TMS
    "num_runs_reduced": 50,  # For computationally intensive algorithms
    "random_state_base": 42
}

# Predefined complete experiment configurations
COMPLETE_EXPERIMENTS = {
    "small_scale_test": {
        "dag_configs": ["gaussian_elimination"],
        "processors": [2, 4],
        "buses": [1, 2],
        "ccr_values": [0.5, 1.0],
        "algorithms": ["cctms", "qlcctms"],
        "num_runs": 10,
        "description": "Small scale test for validation"
    },
    "full_gaussian_experiment": {
        "dag_configs": ["gaussian_elimination"],
        "processors": [2, 4, 6, 8],
        "buses": [1, 2, 3, 4],
        "ccr_values": [0.5, 1.0, 1.5, 2.0],
        "algorithms": ["cctms", "qlcctms"],
        "num_runs": 100,
        "description": "Complete Gaussian Elimination experiment"
    },
    "all_dags_comparison": {
        "dag_configs": ["gaussian_elimination", "epigenomics", "laplace", "stencil"],
        "processors": [2, 4, 6, 8],
        "buses": [1, 2, 3, 4],
        "ccr_values": [0.5, 1.0, 1.5, 2.0],
        "algorithms": ["cctms", "qlcctms"],
        "num_runs": 100,
        "description": "Comprehensive comparison across all DAG types"
    }
}