"""
DAG Generation Module

Contains DAG generators for different benchmark problems:
- Gaussian Elimination
- Epigenomics
- Laplace
- Stencil

Each generator follows the specifications from research papers.
"""

import networkx as nx
import numpy as np
from typing import Tuple, List
from abc import ABC, abstractmethod


class DAGGenerator(ABC):
    """Abstract base class for DAG generators."""
    
    @abstractmethod
    def generate(self, **kwargs) -> Tuple[nx.DiGraph, List[str], List[str]]:
        """Generate a DAG with specified parameters.
        
        Returns:
            Tuple of (graph, task_nodes, message_nodes)
        """
        pass
    
    @abstractmethod
    def get_theoretical_counts(self, **kwargs) -> Tuple[int, int]:
        """Get theoretical task and message counts for validation.
        
        Returns:
            Tuple of (task_count, message_count)
        """
        pass


class GaussianEliminationDAG(DAGGenerator):
    """
    Gaussian Elimination DAG Generator
    
    Theoretical counts:
    - Task nodes: (χ² + χ - 2) / 2
    - Message nodes: χ² - χ - 1
    
    Where χ is the number of linear equations.
    """
    
    def generate(self, chi: int) -> Tuple[nx.DiGraph, List[str], List[str]]:
        """Generate Gaussian Elimination DAG for χ equations."""
        G = nx.DiGraph()
        task_nodes = []
        message_nodes = []
        node_count = 0
        layer = 0
        layers = []
        current_n = chi
        msg_node_count = 0

        # Step 1: Create layers and task nodes
        while current_n > 1:
            # Add pivot node
            pivot_name = f"T{node_count+1}"
            G.add_node(pivot_name, type='task', layer=layer)
            task_nodes.append(pivot_name)
            layers.append([pivot_name])
            node_count += 1
            layer += 1
            
            # Add elimination nodes
            elim_nodes = []
            for i in range(current_n-1):
                name = f"T{node_count+1}"
                G.add_node(name, type='task', layer=layer)
                task_nodes.append(name)
                elim_nodes.append(name)
                node_count += 1
            layers.append(elim_nodes)
            layer += 1
            current_n -= 1

        # Step 2: Add edges with messages
        num_layers = len(layers)
        for l in range(num_layers):
            if l == 0:
                continue
            if l % 2 == 1:
                for node in layers[l]:
                    # all elimination nodes depend on previous pivot
                    msg_name = f"M{msg_node_count+1}"
                    G.add_node(msg_name, type='message')
                    message_nodes.append(msg_name)
                    G.add_edge(layers[l-1][0], msg_name)
                    G.add_edge(msg_name, node)
                    msg_node_count += 1
                    
                # diagonal dependencies to next elimination layer
                if l+2 < num_layers:
                    for j in range(1, len(layers[l])):
                        msg_name = f"M{msg_node_count+1}"
                        G.add_node(msg_name, type='message')
                        message_nodes.append(msg_name)
                        G.add_edge(layers[l][j], msg_name)
                        G.add_edge(msg_name, layers[l+2][j-1])
                        msg_node_count += 1
                        
                # connect first elimination to next pivot
                if l+1 < num_layers:
                    msg_name = f"M{msg_node_count+1}"
                    G.add_node(msg_name, type='message')
                    message_nodes.append(msg_name)
                    G.add_edge(layers[l][0], msg_name)
                    G.add_edge(msg_name, layers[l+1][0])
                    msg_node_count += 1

        return G, task_nodes, message_nodes
    
    def get_theoretical_counts(self, chi: int) -> Tuple[int, int]:
        """Get theoretical task and message counts."""
        task_count = (chi**2 + chi - 2) // 2
        message_count = chi**2 - chi - 1
        return task_count, message_count


class EpigenomicsDAG(DAGGenerator):
    """
    Epigenomics DAG Generator
    
    Theoretical counts:
    - Task nodes: 4γ + 4
    - Message nodes: 5γ + 2
    
    Where γ is the number of parallel branches.
    """
    
    def generate(self, gamma: int) -> Tuple[nx.DiGraph, List[str], List[str]]:
        """Generate Epigenomics DAG for γ parallel branches."""
        n_nodes = 4 * gamma + 4
        tasks = {i: {"id": i, "name": f"T{i+1}", "preds": [], "succs": []} for i in range(n_nodes)}
        
        # Build the epigenomics structure
        initial_tasks = [i for i in range(1, gamma+1)]
        tasks[0]["succs"].extend(initial_tasks)
        
        for i in range(1, gamma+1):
            tasks[i]["preds"].append(0)
            for j in range(4):
                idx = i + (j+1)*gamma
                from_idx = i + j*gamma
                if idx < n_nodes:
                    tasks[idx]["preds"].append(from_idx)
            for j in range(3):
                from_idx = i + j*gamma
                to_idx = i + (j+1)*gamma
                if to_idx < n_nodes:
                    tasks[from_idx]["succs"].append(to_idx)
            if i + 3*gamma < n_nodes and 4*gamma+1 < n_nodes:
                tasks[i+3*gamma]["succs"].append(4*gamma+1)
                tasks[4*gamma+1]["preds"].append(i+3*gamma)
                
        for j in range(4*gamma+1, n_nodes-1):
            tasks[j]["succs"].append(j+1)
            tasks[j+1]["preds"].append(j)
        
        # Back-propagate predecessors
        for t in tasks.values():
            for s in t["succs"]:
                if t["id"] not in tasks[s]["preds"]:
                    tasks[s]["preds"].append(t["id"])

        # Create dependencies and messages
        dependencies = []
        for t in tasks.values():
            for s in t["succs"]:
                dependencies.append((t["id"], s))

        messages = {}
        for mid, (src, dst) in enumerate(dependencies):
            messages[mid] = {"id": mid, "name": f"M{mid+1}", "src": src, "dst": dst}

        # Build networkx graph
        G = nx.DiGraph()
        for t in tasks.values():
            G.add_node(t["name"], type="task", id=t["id"])
        for m in messages.values():
            G.add_node(m["name"], type="message", id=m["id"])
            src_name = f"T{m['src']+1}"
            dst_name = f"T{m['dst']+1}"
            G.add_edge(src_name, m["name"])
            G.add_edge(m["name"], dst_name)

        task_nodes = [f"T{i+1}" for i in range(n_nodes)]
        message_nodes = [f"M{i+1}" for i in range(len(messages))]
        return G, task_nodes, message_nodes
    
    def get_theoretical_counts(self, gamma: int) -> Tuple[int, int]:
        """Get theoretical task and message counts."""
        task_count = 4 * gamma + 4
        message_count = 5 * gamma + 2
        return task_count, message_count


class LaplaceDAG(DAGGenerator):
    """
    Laplace DAG Generator
    
    Theoretical counts:
    - Task nodes: ϕ²
    - Message nodes: 2ϕ² - 2ϕ
    
    Where ϕ is the matrix size.
    """
    
    def generate(self, phi: int) -> Tuple[nx.DiGraph, List[str], List[str]]:
        """Generate Laplace DAG for ϕ×ϕ matrix."""
        n_nodes = phi**2
        tasks = {i: {"id": i, "name": f"T{i+1}", "preds": [], "succs": []} for i in range(n_nodes)}
        current_id = 0
        lllt = 0
        level_sizes1 = list(range(1, phi))
        level_sizes2 = list(range(phi, 1, -1))

        # Build tree-like structure
        for level in level_sizes1:
            count = 1
            for _ in range(level):
                child1, child2 = lllt + count, lllt + count + 1
                tasks[current_id]["succs"].extend([child1, child2])
                count += 1
                current_id += 1
            lllt = child2

        zero = phi
        for level in level_sizes2:
            count = 1
            zero -= 1
            for _ in range(level):
                child1, child2 = lllt + count, lllt + count + 1
                if current_id == lllt-zero or current_id == lllt:
                    tasks[current_id]["succs"].append(child1)
                else:
                    tasks[current_id]["succs"].extend([child1, child2])
                    count += 1
                current_id += 1
            lllt = child2-1

        # Back-propagate predecessors
        for t in tasks.values():
            for s in t["succs"]:
                tasks[s]["preds"].append(t["id"])

        # Create dependencies and messages
        dependencies = []
        for t in tasks.values():
            for s in t["succs"]:
                dependencies.append((t["id"], s))

        messages = {}
        for mid, (src, dst) in enumerate(dependencies):
            messages[mid] = {"id": mid, "name": f"M{mid+1}", "src": src, "dst": dst}

        # Build networkx graph
        G = nx.DiGraph()
        for t in tasks.values():
            G.add_node(t["name"], type="task", id=t["id"])
        for m in messages.values():
            G.add_node(m["name"], type="message", id=m["id"])
            src_name = f"T{m['src']+1}"
            dst_name = f"T{m['dst']+1}"
            G.add_edge(src_name, m["name"])
            G.add_edge(m["name"], dst_name)

        task_nodes = [f"T{i+1}" for i in range(n_nodes)]
        message_nodes = [f"M{i+1}" for i in range(len(messages))]
        return G, task_nodes, message_nodes
    
    def get_theoretical_counts(self, phi: int) -> Tuple[int, int]:
        """Get theoretical task and message counts."""
        task_count = phi**2
        message_count = 2 * phi**2 - 2 * phi
        return task_count, message_count


class StencilDAG(DAGGenerator):
    """
    Stencil DAG Generator
    
    Theoretical counts:
    - Task nodes: λ × ξ
    - Message nodes: (λ - 1) × (3ξ - 2)
    
    Where λ is levels and ξ is tasks per level.
    For simplicity, we assume λ = ξ.
    """
    
    def generate(self, xi: int) -> Tuple[nx.DiGraph, List[str], List[str]]:
        """Generate Stencil DAG with ξ levels and ξ tasks per level."""
        G = nx.DiGraph()
        task_nodes = []
        message_nodes = []
        node_count = 0
        layers = []
        msg_node_count = 0

        # Create layers
        for i in range(xi):
            layer = []
            for j in range(xi):
                name = f"T{node_count+1}"
                G.add_node(name, type='task', layer=i)
                task_nodes.append(name)
                node_count += 1
                layer.append(name)
            layers.append(layer)

        # Add dependencies with messages
        for l in range(1, xi):
            for i in range(xi):
                for j in range(max(0, i-1), min(i+2, xi)):
                    msg_name = f"M{msg_node_count+1}"
                    G.add_node(msg_name, type='message')
                    message_nodes.append(msg_name)
                    G.add_edge(layers[l-1][j], msg_name)
                    G.add_edge(msg_name, layers[l][i])
                    msg_node_count += 1

        return G, task_nodes, message_nodes
    
    def get_theoretical_counts(self, xi: int) -> Tuple[int, int]:
        """Get theoretical task and message counts."""
        task_count = xi * xi  # λ = ξ
        message_count = (xi - 1) * (3 * xi - 2)
        return task_count, message_count


# Factory for creating DAG generators
class DAGFactory:
    """Factory for creating DAG generators."""
    
    _generators = {
        'gaussian': GaussianEliminationDAG,
        'epigenomics': EpigenomicsDAG,
        'laplace': LaplaceDAG,
        'stencil': StencilDAG
    }
    
    @classmethod
    def create_generator(cls, dag_type: str) -> DAGGenerator:
        """Create a DAG generator of the specified type."""
        if dag_type not in cls._generators:
            raise ValueError(f"Unknown DAG type: {dag_type}. Available: {list(cls._generators.keys())}")
        return cls._generators[dag_type]()
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available DAG types."""
        return list(cls._generators.keys())