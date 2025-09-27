"""
Custom Graph Test Script

This script demonstrates how to test the framework with a custom DAG
instead of using the predefined benchmark DAGs.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx
import numpy as np
from src.schedulers import SchedulerFactory
from src.visualization import DAGVisualizer, ScheduleVisualizer


def create_custom_graph():
    """Create the custom graph as specified by the user."""
    
    # Define the edges
    E = [('T1', 'M1'), ('T1', 'M2'), ('T1', 'M3'),
         ('M1', 'T2'), ('M2', 'T3'), ('M3', 'T4'),
         ('T2', 'M4'), ('T3', 'M5'), ('T4', 'M6'),
         ('M4', 'T5'), ('M5', 'T5'), ('M6', 'T6'),
         ('T5', 'M7'), ('M7', 'T6')]
    
    # Create the graph
    graph = nx.DiGraph(E)
    
    # Add node types
    for node in graph.nodes:
        if node.startswith("T"):
            graph.nodes[node]['type'] = 'task'
        else:
            graph.nodes[node]['type'] = 'message'
    
    return graph


def create_custom_cost_matrices():
    """Create the custom cost matrices as specified by the user."""
    
    # Execution Time matrix: ET[i][j] = cost of computation of ith task on jth processor
    ET = np.array([
        [4, 3],  # T1
        [8, 5],  # T2
        [3, 4],  # T3
        [2, 3],  # T4
        [4, 2],  # T5
        [2, 3]   # T6
    ])
    
    # Communication Time matrix: CT[i][j] = cost of emitting message i on bus j
    CT = np.array([
        [2, 3],  # M1
        [2, 3],  # M2
        [5, 3],  # M3
        [3, 4],  # M4
        [3, 2],  # M5
        [1, 3],  # M6
        [3, 2]   # M7
    ])
    
    # Task and message lists
    task_list = [f"T{i}" for i in range(1, 7)]
    message_list = [f"M{i}" for i in range(1, 8)]
    
    return ET, CT, task_list, message_list


def test_custom_graph():
    """Test the framework with the custom graph."""
    
    print("=" * 70)
    print("CUSTOM GRAPH TESTING")
    print("=" * 70)
    
    # Create the custom graph and cost matrices
    graph = create_custom_graph()
    ET, CT, task_list, message_list = create_custom_cost_matrices()
    
    print(f"Custom graph created:")
    print(f"  Tasks: {task_list}")
    print(f"  Messages: {message_list}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  Processors: {ET.shape[1]}")
    print(f"  Buses: {CT.shape[1]}")
    
    # Validate the graph structure
    print(f"\nGraph validation:")
    print(f"  Is DAG: {nx.is_directed_acyclic_graph(graph)}")
    print(f"  Task nodes: {[n for n in graph.nodes if graph.nodes[n]['type'] == 'task']}")
    print(f"  Message nodes: {[n for n in graph.nodes if graph.nodes[n]['type'] == 'message']}")
    
    # Display cost matrices
    print(f"\nExecution Time Matrix (ET):")
    print("Tasks  P0  P1")
    for i, task in enumerate(task_list):
        print(f"{task:4s}: {ET[i][0]:2d}  {ET[i][1]:2d}")
    
    print(f"\nCommunication Time Matrix (CT):")
    print("Msgs  B0  B1")
    for i, msg in enumerate(message_list):
        print(f"{msg:4s}: {CT[i][0]:2d}  {CT[i][1]:2d}")
    
    # Test both scheduling algorithms
    algorithms = ['cctms', 'qlcctms']
    results = {}
    
    print(f"\n" + "=" * 70)
    print("SCHEDULING ALGORITHM TESTING")
    print("=" * 70)
    
    for alg in algorithms:
        print(f"\nTesting {alg.upper()} algorithm...")
        
        # Create scheduler
        if alg == 'qlcctms':
            # Use fewer episodes for quick testing
            scheduler = SchedulerFactory.create_scheduler(alg, max_episodes=10000)
        else:
            scheduler = SchedulerFactory.create_scheduler(alg)
        
        # Run scheduling
        result = scheduler.schedule(
            graph, task_list, message_list, ET, CT, random_state=42
        )
        
        results[alg] = result
        
        # Display results
        print(f"Results for {alg.upper()}:")
        print(f"  Makespan: {result['makespan']:.2f} ms")
        
        if 'q_learning_episodes' in result:
            print(f"  Q-Learning episodes: {result['q_learning_episodes']}")
            print(f"  Q-Learning converged: {result['q_learning_converged']}")
        
        print(f"  Task assignments:")
        for task in task_list:
            proc = result['task_assignment'][task]
            start = result['node_start'][task]
            finish = result['node_finish'][task]
            print(f"    {task}: Processor {proc}, Start={start:.1f}, Finish={finish:.1f}")
        
        print(f"  Message assignments:")
        for msg in message_list:
            bus = result['msg_assignment'].get(msg, 'Same-Proc')
            start = result['node_start'][msg]
            finish = result['node_finish'][msg]
            if bus != 'Same-Proc':
                print(f"    {msg}: Bus {bus}, Start={start:.1f}, Finish={finish:.1f}")
            else:
                print(f"    {msg}: {bus}, Start={start:.1f}, Finish={finish:.1f}")
    
    # Compare algorithms
    print(f"\n" + "=" * 70)
    print("ALGORITHM COMPARISON")
    print("=" * 70)
    
    if len(results) == 2:
        cctms_makespan = results['cctms']['makespan']
        qlcctms_makespan = results['qlcctms']['makespan']
        ratio = (cctms_makespan / qlcctms_makespan) * 100
        
        print(f"CC-TMS Makespan: {cctms_makespan:.2f} ms")
        print(f"QL-CC-TMS Makespan: {qlcctms_makespan:.2f} ms")
        print(f"Makespan Ratio (CC-TMS/QL-CC-TMS × 100): {ratio:.2f}%")
        
        if ratio > 100:
            improvement = ratio - 100
            print(f"→ QL-CC-TMS performs {improvement:.1f}% better than CC-TMS")
        elif ratio < 100:
            improvement = 100 - ratio
            print(f"→ CC-TMS performs {improvement:.1f}% better than QL-CC-TMS")
        else:
            print(f"→ Both algorithms achieve the same makespan")
    
    # Visualization (optional - comment out if no display available)
    print(f"\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)
    
    try:
        print("Generating DAG visualization...")
        DAGVisualizer.visualize_dag(graph, "Custom Test DAG", figsize=(10, 6))
        
        print("Generating schedule visualizations...")
        for alg, result in results.items():
            ScheduleVisualizer.plot_gantt_chart(
                result, task_list, message_list, 
                f"{alg.upper()} Schedule - Custom DAG",
                figsize=(12, 6)
            )
    
    except Exception as e:
        print(f"Visualization skipped: {e}")
        print("(This is normal if running without display or missing matplotlib)")
    
    return results


def validate_custom_graph():
    """Validate that the custom graph is properly constructed."""
    
    graph = create_custom_graph()
    ET, CT, task_list, message_list = create_custom_cost_matrices()
    
    print("=" * 70)
    print("CUSTOM GRAPH VALIDATION")
    print("=" * 70)
    
    # Check graph properties
    issues = []
    
    # 1. Check if it's a DAG
    if not nx.is_directed_acyclic_graph(graph):
        issues.append("Graph is not a DAG (contains cycles)")
    
    # 2. Check node types
    task_nodes = [n for n in graph.nodes if graph.nodes[n]['type'] == 'task']
    message_nodes = [n for n in graph.nodes if graph.nodes[n]['type'] == 'message']
    
    if set(task_nodes) != set(task_list):
        issues.append(f"Task nodes mismatch: {set(task_nodes)} vs {set(task_list)}")
    
    if set(message_nodes) != set(message_list):
        issues.append(f"Message nodes mismatch: {set(message_nodes)} vs {set(message_list)}")
    
    # 3. Check matrix dimensions
    if ET.shape[0] != len(task_list):
        issues.append(f"ET matrix has {ET.shape[0]} rows but {len(task_list)} tasks")
    
    if CT.shape[0] != len(message_list):
        issues.append(f"CT matrix has {CT.shape[0]} rows but {len(message_list)} messages")
    
    # 4. Check graph structure (task-message-task pattern)
    for edge in graph.edges:
        src, dst = edge
        src_type = graph.nodes[src]['type']
        dst_type = graph.nodes[dst]['type']
        
        # Valid patterns: task->message, message->task
        if src_type == dst_type:
            issues.append(f"Invalid edge {src}->{dst}: {src_type}->{dst_type}")
    
    # 5. Check that each message has exactly one predecessor and one successor
    for msg in message_nodes:
        predecessors = list(graph.predecessors(msg))
        successors = list(graph.successors(msg))
        
        if len(predecessors) != 1:
            issues.append(f"Message {msg} has {len(predecessors)} predecessors (should be 1)")
        
        if len(successors) != 1:
            issues.append(f"Message {msg} has {len(successors)} successors (should be 1)")
    
    # Report validation results
    if not issues:
        print("✓ Custom graph validation PASSED")
        print("  Graph structure is valid for the scheduling framework")
    else:
        print("✗ Custom graph validation FAILED")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    # Display graph statistics
    print(f"\nGraph Statistics:")
    print(f"  Total nodes: {len(graph.nodes)}")
    print(f"  Task nodes: {len(task_nodes)}")
    print(f"  Message nodes: {len(message_nodes)}")
    print(f"  Total edges: {len(graph.edges)}")
    print(f"  Processors: {ET.shape[1]}")
    print(f"  Buses: {CT.shape[1]}")
    
    # Find entry and exit tasks
    entry_tasks = [n for n in task_nodes if graph.in_degree(n) == 0]
    exit_tasks = [n for n in task_nodes if graph.out_degree(n) == 0]
    
    print(f"  Entry tasks: {entry_tasks}")
    print(f"  Exit tasks: {exit_tasks}")
    
    return True


if __name__ == "__main__":
    print("DAG Scheduling Framework - Custom Graph Test")
    print("=" * 70)
    
    # First validate the custom graph
    if validate_custom_graph():
        print("\n")
        # Then test the scheduling algorithms
        results = test_custom_graph()
        
        print(f"\n" + "=" * 70)
        print("CUSTOM GRAPH TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nYour custom graph has been successfully tested with both algorithms.")
        print("You can modify this script to test different graphs or cost matrices.")
        
    else:
        print("\n" + "=" * 70)
        print("CUSTOM GRAPH TEST FAILED!")
        print("=" * 70)
        print("Please fix the validation issues above before running the test.")