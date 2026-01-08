"""
Visualization utilities for Air Quality AQI project
Provides plotting functions for semi-supervised learning analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# AQI Classes and colors
AQI_CLASSES = ["Good", "Moderate", "Unhealthy_for_Sensitive_Groups", 
               "Unhealthy", "Very_Unhealthy", "Hazardous"]

AQI_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy_for_Sensitive_Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very_Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023"
}


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap
    
    Parameters:
    -----------
    cm : np.ndarray
        Confusion matrix
    classes : List[str]
        Class labels
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    save_path : Optional[str]
        Path to save figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, 
                ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_self_training_progress(
    history: Dict,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot self-training iteration progress
    
    Parameters:
    -----------
    history : Dict
        Training history with keys: iteration, n_added, val_accuracy, val_f1_macro
    figsize : Tuple[int, int]
        Figure size
    save_path : Optional[str]
        Path to save figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    iterations = history.get('iteration', [])
    
    # Plot 1: Pseudo-labels added
    if 'n_added' in history:
        axes[0].plot(iterations, history['n_added'], 
                    marker='o', linewidth=2.5, markersize=8, 
                    color='#2ca02c', label='Pseudo-labels')
        axes[0].set_xlabel('Iteration', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        axes[0].set_title('Pseudo-labels Added per Iteration', 
                         fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    
    # Plot 2: Validation Accuracy
    if 'val_accuracy' in history:
        axes[1].plot(iterations, history['val_accuracy'], 
                    marker='s', linewidth=2.5, markersize=8,
                    color='#1f77b4', label='Validation Accuracy')
        axes[1].set_xlabel('Iteration', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_title('Validation Accuracy Evolution', 
                         fontsize=13, fontweight='bold')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    # Plot 3: Validation F1-Macro
    if 'val_f1_macro' in history:
        axes[2].plot(iterations, history['val_f1_macro'], 
                    marker='^', linewidth=2.5, markersize=8,
                    color='#ff7f0e', label='Validation F1-Macro')
        axes[2].set_xlabel('Iteration', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('F1-Macro Score', fontsize=12, fontweight='bold')
        axes[2].set_title('Validation F1-Macro Evolution', 
                         fontsize=13, fontweight='bold')
        axes[2].set_ylim([0, 1])
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cotraining_progress(
    history: Dict,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot co-training iteration progress for both models
    
    Parameters:
    -----------
    history : Dict
        Training history with keys for both models
    figsize : Tuple[int, int]
        Figure size
    save_path : Optional[str]
        Path to save figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    iterations = history.get('iteration', [])
    
    # Plot 1: Pseudo-labels exchange
    ax = axes[0, 0]
    if 'n_added_m1' in history and 'n_added_m2' in history:
        ax.plot(iterations, history['n_added_m1'], 
               marker='o', linewidth=2.5, markersize=8,
               label='M1 → M2', color='#1f77b4')
        ax.plot(iterations, history['n_added_m2'], 
               marker='s', linewidth=2.5, markersize=8,
               label='M2 → M1', color='#ff7f0e')
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pseudo-labels Added', fontsize=12, fontweight='bold')
        ax.set_title('Pseudo-label Exchange Between Models', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy comparison
    ax = axes[0, 1]
    if 'val_accuracy_m1' in history and 'val_accuracy_m2' in history:
        ax.plot(iterations, history['val_accuracy_m1'], 
               marker='o', linewidth=2.5, markersize=8,
               label='Model 1', color='#1f77b4')
        ax.plot(iterations, history['val_accuracy_m2'], 
               marker='s', linewidth=2.5, markersize=8,
               label='Model 2', color='#ff7f0e')
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Validation Accuracy: Both Models', 
                    fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Validation F1-Macro comparison
    ax = axes[1, 0]
    if 'val_f1_macro_m1' in history and 'val_f1_macro_m2' in history:
        ax.plot(iterations, history['val_f1_macro_m1'], 
               marker='o', linewidth=2.5, markersize=8,
               label='Model 1', color='#1f77b4')
        ax.plot(iterations, history['val_f1_macro_m2'], 
               marker='s', linewidth=2.5, markersize=8,
               label='Model 2', color='#ff7f0e')
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Validation F1-Macro', fontsize=12, fontweight='bold')
        ax.set_title('Validation F1-Macro: Both Models', 
                    fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative pseudo-labels
    ax = axes[1, 1]
    if 'n_added_m1' in history and 'n_added_m2' in history:
        cumsum_m1 = np.cumsum(history['n_added_m1'])
        cumsum_m2 = np.cumsum(history['n_added_m2'])
        ax.plot(iterations, cumsum_m1, 
               marker='o', linewidth=2.5, markersize=8,
               label='M1 → M2 (Cumulative)', color='#1f77b4')
        ax.plot(iterations, cumsum_m2, 
               marker='s', linewidth=2.5, markersize=8,
               label='M2 → M1 (Cumulative)', color='#ff7f0e')
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Pseudo-labels', fontsize=12, fontweight='bold')
        ax.set_title('Cumulative Pseudo-labeling Progress', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_method_comparison(
    metrics_dict: Dict[str, Dict],
    metric_names: List[str] = ['test_accuracy', 'test_f1_macro'],
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare metrics across different methods
    
    Parameters:
    -----------
    metrics_dict : Dict[str, Dict]
        Dictionary with method names as keys and metrics dict as values
    metric_names : List[str]
        List of metric names to compare
    figsize : Tuple[int, int]
        Figure size
    save_path : Optional[str]
        Path to save figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    methods = list(metrics_dict.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        values = [metrics_dict[m].get(metric_name, 0) for m in methods]
        
        bars = ax.bar(methods, values, color=colors[:len(methods)], alpha=0.8)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), 
                     fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name.replace("_", " ").title()}', 
                    fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xticklabels(methods, rotation=15, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_class_performance(
    classification_report: Dict,
    classes: List[str] = AQI_CLASSES,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot per-class performance metrics
    
    Parameters:
    -----------
    classification_report : Dict
        Sklearn classification report as dict
    classes : List[str]
        Class names
    figsize : Tuple[int, int]
        Figure size
    save_path : Optional[str]
        Path to save figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    # Extract per-class metrics
    metrics = []
    for cls in classes:
        if cls in classification_report:
            metrics.append({
                'Class': cls,
                'Precision': classification_report[cls]['precision'],
                'Recall': classification_report[cls]['recall'],
                'F1-Score': classification_report[cls]['f1-score']
            })
    
    if not metrics:
        return None
    
    df = pd.DataFrame(metrics)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(df))
    width = 0.25
    
    bars1 = ax.bar(x - width, df['Precision'], width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, df['Recall'], width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, df['F1-Score'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('AQI Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Class'], rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_alert_analysis(
    predictions_df: pd.DataFrame,
    station_col: str = 'station',
    alert_col: str = 'is_alert',
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot alert analysis by station
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        Predictions dataframe with station and alert columns
    station_col : str
        Column name for station
    alert_col : str
        Column name for alert flag
    figsize : Tuple[int, int]
        Figure size
    save_path : Optional[str]
        Path to save figure
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    if station_col not in predictions_df.columns or alert_col not in predictions_df.columns:
        return None
    
    # Calculate alert rate by station
    station_stats = predictions_df.groupby(station_col)[alert_col].agg(['sum', 'count'])
    station_stats['alert_rate'] = (station_stats['sum'] / station_stats['count'] * 100)
    station_stats = station_stats.sort_values('alert_rate', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Alert counts
    stations = station_stats.index
    ax1.barh(stations, station_stats['sum'], color='#d62728', alpha=0.8)
    ax1.set_xlabel('Number of Alerts', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Station', fontsize=12, fontweight='bold')
    ax1.set_title('Alert Count by Station', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Alert rate
    ax2.barh(stations, station_stats['alert_rate'], color='#ff7f0e', alpha=0.8)
    ax2.set_xlabel('Alert Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Station', fontsize=12, fontweight='bold')
    ax2.set_title('Alert Rate by Station', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_report_summary(
    baseline_metrics: Dict,
    self_training_metrics: Dict,
    cotraining_metrics: Dict,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create summary comparison table
    
    Parameters:
    -----------
    baseline_metrics : Dict
        Baseline metrics
    self_training_metrics : Dict
        Self-training metrics
    cotraining_metrics : Dict
        Co-training metrics
    save_path : Optional[str]
        Path to save CSV
        
    Returns:
    --------
    df : pd.DataFrame
    """
    summary = {
        'Method': ['Baseline', 'Self-Training', 'Co-Training'],
        'Test Accuracy': [
            baseline_metrics.get('test_accuracy', 0),
            self_training_metrics.get('test_accuracy', 0),
            cotraining_metrics.get('test_accuracy', 0)
        ],
        'Test F1-Macro': [
            baseline_metrics.get('test_f1_macro', 0),
            self_training_metrics.get('test_f1_macro', 0),
            cotraining_metrics.get('test_f1_macro', 0)
        ]
    }
    
    df = pd.DataFrame(summary)
    
    # Calculate improvement
    df['Accuracy Improvement'] = df['Test Accuracy'] - df.loc[0, 'Test Accuracy']
    df['F1 Improvement'] = df['Test F1-Macro'] - df.loc[0, 'Test F1-Macro']
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df
