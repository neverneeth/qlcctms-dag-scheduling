"""
Cost Matrix Generation Module

Generates execution time and communication time matrices for DAGs
based on experimental parameters.
"""

import numpy as np
import networkx as nx
from typing import Tuple, List, Dict


def generate_cost_matrices(graph: nx.DiGraph, num_processors: int, num_buses: int,
                          ccr: float = 1.0, et_min: float = 10, et_max: float = 30,
                          ct_min: float = 10, ct_max: float = 30,
                          random_state: int = None) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Generate execution time and communication time matrices.
    
    Args:
        graph: DAG with task and message nodes
        num_processors: Number of processors
        num_buses: Number of buses
        ccr: Communication-to-Computation Ratio
        et_min: Minimum execution time (ms)
        et_max: Maximum execution time (ms)
        ct_min: Minimum communication time (ms)
        ct_max: Maximum communication time (ms)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (ET_matrix, CT_matrix, task_list, message_list)
        - ET_matrix: [num_tasks x num_processors] execution times
        - CT_matrix: [num_messages x num_buses] communication times
        - task_list: List of task node names
        - message_list: List of message node names
    """
    rng = np.random.default_rng(random_state)
    
    # Extract task and message nodes
    task_list = [n for n, d in graph.nodes(data=True) if d.get('type') == 'task']
    message_list = [n for n, d in graph.nodes(data=True) if d.get('type') == 'message']
    
    num_tasks = len(task_list)
    num_messages = len(message_list)
    
    # Generate execution times (uniform distribution)
    ET = rng.uniform(et_min, et_max, size=(num_tasks, num_processors))
    
    # Generate communication times (uniform distribution)
    CT = rng.uniform(ct_min, ct_max, size=(num_messages, num_buses))
    
    # Scale communication times to achieve desired CCR
    if num_messages > 0 and ccr > 0:
        mean_ET = np.mean(ET)
        mean_CT = np.mean(CT)
        if mean_CT > 0:
            scale_factor = ccr * mean_ET / mean_CT
            CT *= scale_factor
    
    return ET, CT, task_list, message_list


def validate_cost_matrices(ET: np.ndarray, CT: np.ndarray, task_list: List[str], 
                         message_list: List[str]) -> Dict[str, any]:
    """
    Validate and provide statistics for generated cost matrices.
    
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'num_tasks': len(task_list),
        'num_messages': len(message_list),
        'num_processors': ET.shape[1] if ET.size > 0 else 0,
        'num_buses': CT.shape[1] if CT.size > 0 else 0,
        'et_min': np.min(ET) if ET.size > 0 else 0,
        'et_max': np.max(ET) if ET.size > 0 else 0,
        'et_mean': np.mean(ET) if ET.size > 0 else 0,
        'ct_min': np.min(CT) if CT.size > 0 else 0,
        'ct_max': np.max(CT) if CT.size > 0 else 0,
        'ct_mean': np.mean(CT) if CT.size > 0 else 0,
        'actual_ccr': np.mean(CT) / np.mean(ET) if ET.size > 0 and np.mean(ET) > 0 else 0
    }
    
    return stats