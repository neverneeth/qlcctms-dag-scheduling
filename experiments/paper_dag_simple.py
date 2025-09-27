"""
Simple Custom Graph Integration Example

This shows the minimal steps to test your custom graph with the framework.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx
import numpy as np
from src.schedulers import SchedulerFactory

def main():
    # Step 1: Create your custom graph exactly as you defined it
    E = [('T1', 'M1'), ('T1', 'M2'), ('T1', 'M3'),
         ('M1', 'T2'), ('M2', 'T3'), ('M3', 'T4'),
         ('T2', 'M4'), ('T3', 'M5'), ('T4', 'M6'),
         ('M4', 'T5'), ('M5', 'T5'), ('M6', 'T6'),
         ('T5', 'M7'), ('M7', 'T6')]
    
    graph = nx.DiGraph(E)
    
    # Step 2: Add node types
    for node in graph.nodes:
        if node.startswith("T"):
            graph.nodes[node]['type'] = 'task'
        else:
            graph.nodes[node]['type'] = 'message'
    
    # Step 3: Define your cost matrices
    ET = np.array([[4, 3], [8, 5], [3, 4], [2, 3], [4, 2], [2, 3]])  # 6 tasks x 2 processors
    CT = np.array([[2, 3], [2, 3], [5, 3], [3, 4], [3, 2], [1, 3], [3, 2]])  # 7 messages x 2 buses
    
    task_list = [f"T{i}" for i in range(1, 7)]
    message_list = [f"M{i}" for i in range(1, 8)]
    
    # Step 4: Test with CC-TMS algorithm
    print("Testing CC-TMS algorithm...")
    cctms_scheduler = SchedulerFactory.create_scheduler('cctms')
    cctms_result = cctms_scheduler.schedule(graph, task_list, message_list, ET, CT, random_state=42)
    
    print(f"CC-TMS Makespan: {cctms_result['makespan']:.2f} ms")
    print("Task assignments:")
    for task in task_list:
        proc = cctms_result['task_assignment'][task]
        start = cctms_result['node_start'][task]
        finish = cctms_result['node_finish'][task]
        print(f"  {task}: Processor {proc}, Start={start:.1f}, Finish={finish:.1f}")
    
    # Step 5: Test with QL-CC-TMS algorithm
    print("\nTesting QL-CC-TMS algorithm...")
    qlcctms_scheduler = SchedulerFactory.create_scheduler('qlcctms', max_episodes=5000)
    qlcctms_result = qlcctms_scheduler.schedule(graph, task_list, message_list, ET, CT, random_state=42)
    
    print(f"QL-CC-TMS Makespan: {qlcctms_result['makespan']:.2f} ms")
    print(f"Q-Learning converged: {qlcctms_result['q_learning_converged']}")
    print("Task assignments:")
    for task in task_list:
        proc = qlcctms_result['task_assignment'][task]
        start = qlcctms_result['node_start'][task]
        finish = qlcctms_result['node_finish'][task]
        print(f"  {task}: Processor {proc}, Start={start:.1f}, Finish={finish:.1f}")
    
    # Step 6: Compare results
    ratio = (cctms_result['makespan'] / qlcctms_result['makespan']) * 100
    print(f"\nMakespan Ratio (CC-TMS/QL-CC-TMS × 100): {ratio:.2f}%")
    
    if ratio > 100:
        improvement = ratio - 100
        print(f"QL-CC-TMS performs {improvement:.1f}% better than CC-TMS")
    elif ratio < 100:
        improvement = 100 - ratio
        print(f"CC-TMS performs {improvement:.1f}% better than QL-CC-TMS")
    else:
        print(f"Both algorithms achieve the same makespan")

if __name__ == "__main__":
    main()