"""
Scheduling Algorithms Module

Contains implementations of scheduling algorithms:
- CC-TMS (Communication-Conscious Task and Message Scheduling)
- QL-CC-TMS (Q-Learning based CC-TMS)
"""

import networkx as nx
import numpy as np
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional


class SchedulingAlgorithm(ABC):
    """Abstract base class for scheduling algorithms."""
    
    @abstractmethod
    def schedule(self, graph: nx.DiGraph, task_list: List[str], message_list: List[str], 
                ET: np.ndarray, CT: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Schedule tasks and messages on processors and buses.
        
        Args:
            graph: The DAG
            task_list: List of task node names
            message_list: List of message node names
            ET: Execution time matrix [tasks x processors]
            CT: Communication time matrix [messages x buses]
            
        Returns:
            Schedule dictionary with assignments and timing
        """
        pass


class CCTMSScheduler(SchedulingAlgorithm):
    """CC-TMS (Communication-Conscious Task and Message Scheduling) Algorithm."""
    
    def __init__(self):
        self.name = "CC-TMS"
    
    def schedule(self, graph: nx.DiGraph, task_list: List[str], message_list: List[str], 
                ET: np.ndarray, CT: np.ndarray, **kwargs) -> Dict[str, Any]:
        """CC-TMS scheduling algorithm."""
        # Calculate upward ranks
        avg_ET, avg_CT = self._calculate_average_costs(task_list, message_list, ET, CT)
        rank_u = self._upward_rank(graph, avg_ET, avg_CT)
        
        # Sort tasks by rank (highest first)
        priority_list = sorted(task_list, key=lambda t: -rank_u[t])
        
        # Schedule using list scheduling with CC-TMS heuristic
        return self._cctms_schedule(graph, task_list, message_list, ET, CT, priority_list)
    
    def _calculate_average_costs(self, task_list: List[str], message_list: List[str], 
                               ET: np.ndarray, CT: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate average execution and communication costs."""
        task_index = {t: i for i, t in enumerate(task_list)}
        message_index = {m: i for i, m in enumerate(message_list)}
        
        avg_ET = {}
        for task in task_list:
            idx = task_index[task]
            avg_ET[task] = np.mean(ET[idx])
            
        avg_CT = {}
        for msg in message_list:
            idx = message_index[msg]
            avg_CT[msg] = np.mean(CT[idx])
            
        return avg_ET, avg_CT
    
    def _upward_rank(self, graph: nx.DiGraph, avg_ET: Dict[str, float], 
                    avg_CT: Dict[str, float]) -> Dict[str, float]:
        """Calculate upward rank for all nodes."""
        rank_u = {}
        for node in reversed(list(nx.topological_sort(graph))):
            node_type = graph.nodes[node]['type']
            if node_type == 'task':
                succ_msgs = [succ for succ in graph.successors(node) 
                           if graph.nodes[succ]['type'] == 'message']
                if not succ_msgs:
                    rank_u[node] = avg_ET[node]
                else:
                    rank_u[node] = avg_ET[node] + max(rank_u[m] for m in succ_msgs)
            else:  # message node
                succ_tasks = [succ for succ in graph.successors(node) 
                            if graph.nodes[succ]['type'] == 'task']
                assert len(succ_tasks) == 1, f"Message {node} should have exactly one successor task"
                rank_u[node] = avg_CT[node] + rank_u[succ_tasks[0]]
        return rank_u
    
    def _cctms_schedule(self, graph: nx.DiGraph, task_list: List[str], message_list: List[str],
                       ET: np.ndarray, CT: np.ndarray, priority_list: List[str]) -> Dict[str, Any]:
        """CC-TMS list scheduling algorithm."""
        num_processors = len(ET[0])
        num_buses = len(CT[0])
        task_index = {t: i for i, t in enumerate(task_list)}
        message_index = {m: i for i, m in enumerate(message_list)}

        # Track availability
        proc_avail = [0] * num_processors
        bus_avail = [0] * num_buses
        node_start = {}
        node_finish = {}
        task_assignment = {}
        msg_assignment = {}
        msg_scheduled = set()

        # Schedule each task
        for task in priority_list:
            best_eft = float('inf')
            best_proc = None
            best_start = None
            best_finish = None
            best_msgs = None

            # Try each processor
            for p in range(num_processors):
                temp_proc_avail = proc_avail[p]
                pred_msgs = [pred for pred in graph.predecessors(task) 
                           if graph.nodes[pred]['type'] == 'message']
                temp_bus_avail = bus_avail[:]
                msg_sched = {}
                pred_msg_finish = []

                # Schedule predecessor messages
                for msg in pred_msgs:
                    if msg in msg_scheduled:
                        pred_msg_finish.append(node_finish[msg])
                        continue

                    pred_task = [pred for pred in graph.predecessors(msg) 
                               if graph.nodes[pred]['type'] == 'task'][0]
                    pred_task_proc = task_assignment.get(pred_task, None)

                    if pred_task_proc == p:
                        # Same processor - no communication needed
                        msg_est = node_finish[pred_task]
                        msg_eft = msg_est
                        msg_sched[msg] = (None, msg_est, msg_eft)
                        pred_msg_finish.append(msg_eft)
                        continue

                    # Different processors - find best bus
                    best_msg_eft = float('inf')
                    best_bus = None
                    best_msg_start = None
                    best_msg_finish = None

                    for b in range(num_buses):
                        msg_est = max(temp_bus_avail[b], node_finish[pred_task])
                        msg_eft = msg_est + CT[message_index[msg]][b]

                        if msg_eft < best_msg_eft:
                            best_msg_eft = msg_eft
                            best_bus = b
                            best_msg_start = msg_est
                            best_msg_finish = msg_eft

                    msg_sched[msg] = (best_bus, best_msg_start, best_msg_finish)
                    if best_bus is not None:
                        temp_bus_avail[best_bus] = best_msg_finish
                    pred_msg_finish.append(best_msg_finish)

                # Calculate task timing
                task_est = max([temp_proc_avail] + pred_msg_finish) if pred_msg_finish else temp_proc_avail
                task_eft = task_est + ET[task_index[task]][p]

                if task_eft < best_eft:
                    best_eft = task_eft
                    best_proc = p
                    best_start = task_est
                    best_finish = task_eft
                    best_msgs = msg_sched

            # Commit best assignment
            proc_avail[best_proc] = best_finish
            node_start[task] = best_start
            node_finish[task] = best_finish
            task_assignment[task] = best_proc

            # Assign predecessor messages
            for msg, (bus, msg_start, msg_finish) in best_msgs.items():
                if msg not in msg_scheduled:
                    node_start[msg] = msg_start
                    node_finish[msg] = msg_finish
                    if bus is not None:
                        bus_avail[bus] = msg_finish
                        msg_assignment[msg] = bus
                    msg_scheduled.add(msg)

        # Handle remaining messages
        for msg in message_list:
            if msg not in node_start:
                node_start[msg] = 0
                node_finish[msg] = 0
                msg_assignment[msg] = 0

        return {
            'algorithm': self.name,
            'task_assignment': task_assignment,
            'msg_assignment': msg_assignment,
            'node_start': node_start,
            'node_finish': node_finish,
            'makespan': max(node_finish[t] for t in task_list) if task_list else 0
        }


class QLCCTMSScheduler(SchedulingAlgorithm):
    """QL-CC-TMS (Q-Learning based CC-TMS) Algorithm."""
    
    def __init__(self, epsilon: float = 0.2, learning_rate: float = 0.1, 
                 discount: float = 0.8, max_episodes: int = 300000,
                 convergence_window: int = 40, convergence_threshold: float = 0.2):
        self.name = "QL-CC-TMS"
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.max_episodes = max_episodes
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
    
    def schedule(self, graph: nx.DiGraph, task_list: List[str], message_list: List[str], 
                ET: np.ndarray, CT: np.ndarray, random_state: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """QL-CC-TMS scheduling algorithm."""
        # Set up random number generator
        rng = np.random.default_rng(random_state)
        if random_state is not None:
            random.seed(random_state)
        
        # Calculate upward ranks
        avg_ET, avg_CT = self._calculate_average_costs(task_list, message_list, ET, CT)
        rank_u = self._upward_rank(graph, avg_ET, avg_CT)
        
        # Run Q-learning
        Q, episodes, converged = self._q_learning(graph, rank_u, rng)
        
        # Extract task order from learned Q-table
        priority_list = self._extract_task_order(graph, Q, rng)
        
        # Schedule using CC-TMS with learned priority
        cctms = CCTMSScheduler()
        result = cctms._cctms_schedule(graph, task_list, message_list, ET, CT, priority_list)
        result['algorithm'] = self.name
        result['q_learning_episodes'] = episodes
        result['q_learning_converged'] = converged
        
        return result
    
    def _calculate_average_costs(self, task_list: List[str], message_list: List[str], 
                               ET: np.ndarray, CT: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate average execution and communication costs."""
        task_index = {t: i for i, t in enumerate(task_list)}
        message_index = {m: i for i, m in enumerate(message_list)}
        
        avg_ET = {}
        for task in task_list:
            idx = task_index[task]
            avg_ET[task] = np.mean(ET[idx])
            
        avg_CT = {}
        for msg in message_list:
            idx = message_index[msg]
            avg_CT[msg] = np.mean(CT[idx])
            
        return avg_ET, avg_CT
    
    def _upward_rank(self, graph: nx.DiGraph, avg_ET: Dict[str, float], 
                    avg_CT: Dict[str, float]) -> Dict[str, float]:
        """Calculate upward rank for all nodes."""
        rank_u = {}
        for node in reversed(list(nx.topological_sort(graph))):
            node_type = graph.nodes[node]['type']
            if node_type == 'task':
                succ_msgs = [succ for succ in graph.successors(node) 
                           if graph.nodes[succ]['type'] == 'message']
                if not succ_msgs:
                    rank_u[node] = avg_ET[node]
                else:
                    rank_u[node] = avg_ET[node] + max(rank_u[m] for m in succ_msgs)
            else:  # message node
                succ_tasks = [succ for succ in graph.successors(node) 
                            if graph.nodes[succ]['type'] == 'task']
                assert len(succ_tasks) == 1, f"Message {node} should have exactly one successor task"
                rank_u[node] = avg_CT[node] + rank_u[succ_tasks[0]]
        return rank_u
    
    def _viable_tasks(self, graph: nx.DiGraph, scheduled: set) -> List[str]:
        """Get list of tasks that can be scheduled next."""
        viable = []
        for t in graph.nodes:
            if graph.nodes[t]['type'] != 'task':
                continue
            
            all_pred_tasks_scheduled = True
            for m in graph.predecessors(t):
                if graph.nodes[m]['type'] != 'message':
                    continue
                pred_tasks = [tp for tp in graph.predecessors(m) 
                            if graph.nodes[tp]['type'] == 'task']
                if not pred_tasks or any(tp not in scheduled for tp in pred_tasks):
                    all_pred_tasks_scheduled = False
                    break
            
            if all_pred_tasks_scheduled and t not in scheduled:
                viable.append(t)
        return viable
    
    def _q_learning_episode(self, graph: nx.DiGraph, Q: defaultdict, rank_u: Dict[str, float], 
                          rng: np.random.Generator) -> float:
        """Run one Q-learning episode."""
        scheduled = set()
        entry_tasks = [t for t in graph.nodes if graph.in_degree(t) == 0]
        
        # Initialize Q-values for entry tasks
        for t in entry_tasks:
            if ('START', t) not in Q:
                Q[('START', t)] = rank_u[t]
        
        last_task = rng.choice(entry_tasks)
        scheduled.add(last_task)
        abs_diffs = []
        
        task_count = len([n for n in graph.nodes if graph.nodes[n]['type'] == 'task'])
        
        while len(scheduled) != task_count:
            candidate_set = self._viable_tasks(graph, scheduled)
            candidate_set = [t for t in candidate_set if t not in scheduled]
            
            if not candidate_set:
                break
            
            # Epsilon-greedy action selection
            if rng.random() < self.epsilon:
                y = rng.choice(candidate_set)
            else:
                q_vals = {a: Q.get((last_task, a), 0) for a in candidate_set}
                max_q = max(q_vals.values())
                best_actions = [a for a, v in q_vals.items() if v == max_q]
                y = rng.choice(best_actions)
            
            # Q-learning update
            reward = rank_u[y]
            old = Q.get((last_task, y), 0)
            scheduled.add(y)
            
            next_candidates = self._viable_tasks(graph, scheduled)
            next_candidates = [t for t in next_candidates if t not in scheduled]
            best = max(Q.get((y, a), 0) for a in next_candidates) if next_candidates else 0
            
            Q[(last_task, y)] = old + self.learning_rate * (reward + self.discount * best - old)
            abs_diffs.append(abs(old - Q[(last_task, y)]))
            last_task = y
        
        return sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0
    
    def _q_learning(self, graph: nx.DiGraph, rank_u: Dict[str, float], 
                   rng: np.random.Generator) -> Tuple[defaultdict, int, bool]:
        """Run Q-learning to learn task priorities."""
        Q = defaultdict(float)
        recent_diffs = []
        converged = False
        episodes = 0
        
        while not converged and episodes < self.max_episodes:
            mean_abs_diff = self._q_learning_episode(graph, Q, rank_u, rng)
            recent_diffs.append(mean_abs_diff)
            
            if len(recent_diffs) > self.convergence_window:
                recent_diffs.pop(0)
            
            if (len(recent_diffs) == self.convergence_window and 
                np.mean(recent_diffs) < self.convergence_threshold):
                converged = True
            
            episodes += 1
        
        return Q, episodes, converged
    
    def _extract_task_order(self, graph: nx.DiGraph, Q: defaultdict, 
                           rng: np.random.Generator) -> List[str]:
        """Extract task execution order from learned Q-table."""
        task_order = []
        scheduled = set()
        entry_tasks = [t for t in graph.nodes if graph.in_degree(t) == 0]
        
        last_task = max(entry_tasks, key=lambda t: Q.get(('START', t), 0))
        task_order.append(last_task)
        scheduled.add(last_task)
        
        task_count = len([n for n in graph.nodes if graph.nodes[n]['type'] == 'task'])
        
        while len(scheduled) != task_count:
            candidates = self._viable_tasks(graph, scheduled)
            candidates = [t for t in candidates if t not in scheduled]
            
            if not candidates:
                break
            
            q_vals = {a: Q.get((last_task, a), 0) for a in candidates}
            max_q = max(q_vals.values())
            best_actions = [a for a, v in q_vals.items() if v == max_q]
            next_task = rng.choice(best_actions)
            
            task_order.append(next_task)
            scheduled.add(next_task)
            last_task = next_task
        
        return task_order


# Factory for creating schedulers
class SchedulerFactory:
    """Factory for creating scheduling algorithms."""
    
    _schedulers = {
        'cctms': CCTMSScheduler,
        'qlcctms': QLCCTMSScheduler
    }
    
    @classmethod
    def create_scheduler(cls, algorithm: str, **kwargs) -> SchedulingAlgorithm:
        """Create a scheduler of the specified type."""
        if algorithm not in cls._schedulers:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(cls._schedulers.keys())}")
        return cls._schedulers[algorithm](**kwargs)
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """Get list of available scheduling algorithms."""
        return list(cls._schedulers.keys())