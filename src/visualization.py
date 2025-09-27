"""
Visualization Module

Provides visualization capabilities for DAGs, schedules, and experiment results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import os


class DAGVisualizer:
    """Visualizes DAG structures."""
    
    @staticmethod
    def visualize_dag(graph: nx.DiGraph, title: str = "DAG Structure", 
                     figsize: Tuple[int, int] = (12, 8), 
                     save_path: Optional[str] = None) -> None:
        """
        Visualize a DAG with tasks and messages.
        
        Args:
            graph: NetworkX DiGraph with 'type' attribute on nodes
            title: Title for the plot
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
        """
        # Create a copy to avoid modifying original
        graph_copy = graph.copy()
        
        # Assign layers for topological layout
        for layer, nodes in enumerate(nx.topological_generations(graph_copy)):
            for node in nodes:
                graph_copy.nodes[node]["layer"] = layer
        
        # Create layout
        pos = nx.multipartite_layout(graph_copy, subset_key="layer", align="horizontal")
        
        # Stretch layout for better visibility
        for node in pos:
            pos[node] = (pos[node][0] * 3, pos[node][1] * 2)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Separate nodes by type
        task_nodes = [n for n, d in graph_copy.nodes(data=True) if d.get('type') == 'task']
        message_nodes = [n for n, d in graph_copy.nodes(data=True) if d.get('type') == 'message']
        
        # Draw task nodes (circles)
        nx.draw_networkx_nodes(graph_copy, pos, nodelist=task_nodes,
                              node_color='lightblue', node_shape='o',
                              node_size=800, ax=ax)
        
        # Draw message nodes (squares)
        nx.draw_networkx_nodes(graph_copy, pos, nodelist=message_nodes,
                              node_color='lightcoral', node_shape='s',
                              node_size=600, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(graph_copy, pos, edge_color='gray',
                              arrows=True, arrowsize=20, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(graph_copy, pos, font_size=10, font_weight='bold', ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Add legend
        task_patch = patches.Patch(color='lightblue', label='Tasks')
        message_patch = patches.Patch(color='lightcoral', label='Messages')
        ax.legend(handles=[task_patch, message_patch], loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class ScheduleVisualizer:
    """Visualizes scheduling results."""
    
    @staticmethod
    def plot_gantt_chart(schedule: Dict[str, Any], task_list: List[str], 
                        message_list: List[str], title: str = "Schedule Gantt Chart",
                        figsize: Tuple[int, int] = (12, 8),
                        save_path: Optional[str] = None) -> None:
        """
        Create a Gantt chart for the schedule.
        
        Args:
            schedule: Schedule dictionary from scheduling algorithm
            task_list: List of task names
            message_list: List of message names
            title: Chart title
            figsize: Figure size
            save_path: Optional path to save the figure
        """
        # Determine number of processors and buses
        num_processors = max(schedule['task_assignment'].values()) + 1
        num_buses = max([v for v in schedule['msg_assignment'].values() 
                        if v is not None] + [0]) + 1
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(task_list) + len(message_list)))
        
        # Plot tasks on processors
        for i, task in enumerate(task_list):
            proc = schedule['task_assignment'][task]
            start = schedule['node_start'][task]
            finish = schedule['node_finish'][task]
            duration = finish - start
            
            ax.barh(y=proc, width=duration, left=start, height=0.6,
                   color=colors[i], edgecolor='black', linewidth=1,
                   label=f'Task {task}' if i < 10 else "")
            
            # Add task label
            ax.text(start + duration/2, proc, task, ha='center', va='center',
                   fontsize=10, fontweight='bold')
        
        # Plot messages on buses (below processors)
        for i, msg in enumerate(message_list):
            bus = schedule['msg_assignment'].get(msg)
            if bus is not None:
                start = schedule['node_start'][msg]
                finish = schedule['node_finish'][msg]
                duration = finish - start
                
                y_pos = num_processors + bus
                ax.barh(y=y_pos, width=duration, left=start, height=0.6,
                       color=colors[len(task_list) + i], edgecolor='black',
                       linewidth=1, alpha=0.7)
                
                # Add message label
                if duration > 0:  # Only label if message has duration
                    ax.text(start + duration/2, y_pos, msg, ha='center', va='center',
                           fontsize=9, fontweight='bold')
        
        # Set labels and title
        y_labels = [f'P{i}' for i in range(num_processors)] + [f'B{j}' for j in range(num_buses)]
        ax.set_yticks(range(num_processors + num_buses))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Resources', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add makespan line
        makespan = schedule['makespan']
        ax.axvline(x=makespan, color='red', linestyle='--', linewidth=2,
                  label=f'Makespan = {makespan:.1f} ms')
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, makespan * 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class ResultsVisualizer:
    """Visualizes experiment results."""
    
    @staticmethod
    def plot_makespan_comparison(results_df: pd.DataFrame, 
                               group_by: str = 'dag_param_value',
                               figsize: Tuple[int, int] = (12, 6),
                               save_path: Optional[str] = None) -> None:
        """
        Plot makespan comparison between algorithms.
        
        Args:
            results_df: DataFrame with experiment results
            group_by: Column to group results by
            figsize: Figure size
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Group and calculate mean makespans
        grouped = results_df.groupby([group_by, 'algorithm'])['makespan'].mean().reset_index()
        
        # Create pivot table for plotting
        pivot_df = grouped.pivot(index=group_by, columns='algorithm', values='makespan')
        
        # Plot
        pivot_df.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Average Makespan Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel(group_by.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Makespan (ms)', fontsize=12)
        ax.legend(title='Algorithm')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_makespan_ratios(ratios_df: pd.DataFrame,
                           group_by: str = 'dag_param_value',
                           figsize: Tuple[int, int] = (12, 6),
                           save_path: Optional[str] = None) -> None:
        """
        Plot makespan ratios (CC-TMS/QL-CC-TMS × 100).
        
        Args:
            ratios_df: DataFrame with makespan ratios
            group_by: Column to group results by
            figsize: Figure size
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate mean ratios
        mean_ratios = ratios_df.groupby(group_by)['makespan_ratio'].agg(['mean', 'std']).reset_index()
        
        # Plot with error bars
        ax.errorbar(mean_ratios[group_by], mean_ratios['mean'], 
                   yerr=mean_ratios['std'], marker='o', capsize=5, capthick=2)
        
        # Add horizontal line at 100% (equal performance)
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
        
        ax.set_title('Makespan Ratio: CC-TMS / QL-CC-TMS × 100', fontsize=14, fontweight='bold')
        ax.set_xlabel(group_by.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Makespan Ratio (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_heatmap(results_df: pd.DataFrame, metric: str = 'makespan',
                    x_col: str = 'num_processors', y_col: str = 'ccr',
                    algorithm: str = 'cctms', figsize: Tuple[int, int] = (10, 6),
                    save_path: Optional[str] = None) -> None:
        """
        Plot heatmap of results.
        
        Args:
            results_df: DataFrame with experiment results
            metric: Metric to visualize
            x_col: Column for x-axis
            y_col: Column for y-axis
            algorithm: Algorithm to filter by
            figsize: Figure size
            save_path: Optional path to save the figure
        """
        # Filter by algorithm
        alg_data = results_df[results_df['algorithm'] == algorithm]
        
        # Calculate mean values
        heatmap_data = alg_data.groupby([y_col, x_col])[metric].mean().reset_index()
        pivot_data = heatmap_data.pivot(index=y_col, columns=x_col, values=metric)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
        
        ax.set_title(f'{metric.title()} Heatmap - {algorithm.upper()}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_execution_time_comparison(results_df: pd.DataFrame,
                                     figsize: Tuple[int, int] = (12, 6),
                                     save_path: Optional[str] = None) -> None:
        """
        Plot execution time comparison between algorithms.
        
        Args:
            results_df: DataFrame with experiment results
            figsize: Figure size
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Box plot of execution times by algorithm
        algorithms = results_df['algorithm'].unique()
        exec_times = [results_df[results_df['algorithm'] == alg]['execution_time'].values 
                     for alg in algorithms]
        
        box_plot = ax.boxplot(exec_times, labels=algorithms, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors[:len(algorithms)]):
            patch.set_facecolor(color)
        
        ax.set_title('Algorithm Execution Time Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()