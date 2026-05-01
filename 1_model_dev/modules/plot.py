import traceback
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional, Dict, Any
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

def plot_LearnCurve(
    data_type: str | list,
    data: pd.DataFrame, 
    out_folder:str
):
    """
    Plot learning curves for train and valid splits.

    Parameters
    ----------
    data_type : str or list of str
        One of: "loss", "acc", "f1", "prec", "sens"
        Or a list of two: e.g. ["acc", "f1"]
    data : pd.DataFrame
        Must contain columns: epoch, train_valid_test, and the relevant metric columns.
        'train_valid_test' should contain "train" and "valid" labels.
    """

    # ------------------------------------------------------------------ #
    # Column name → display label mapping
    # ------------------------------------------------------------------ #
    LABEL_MAP = {
        "loss": "Loss",
        "acc":  "Accuracy",
        "f1":   "F1",
        "prec": "Precision",
        "sens": "Sensitivity",
        "dropout": "Dropout",
    }

    # Marker map for two-data-type mode
    MARKER_MAP = {
        "acc":  "o",   # circle
        "f1":   "^",   # triangle up
        "prec": "s",   # square
        "sens": "D",   # diamond
        "loss": "o",
        "dropout": "o",
    }

    if data_type == "loss":
        legend_loc = "upper right"  
    elif data_type == "dropout":
        legend_loc = "upper left"  
    else:
        legend_loc = "lower right"

    # ------------------------------------------------------------------ #
    # Normalise data_type to a list
    # ------------------------------------------------------------------ #
    if isinstance(data_type, str):
        data_type = [data_type]

    if len(data_type) not in (1, 2):
        raise ValueError("data_type must contain 1 or 2 metric names.")

    # ------------------------------------------------------------------ #
    # Filter to train / valid only and pivot for easy access
    # ------------------------------------------------------------------ #
    try:
        df = data[data["train_valid_test"].isin(["train", "valid"])].copy()

        train_df = df[df["train_valid_test"] == "train"].sort_values("epoch")
        valid_df = df[df["train_valid_test"] == "valid"].sort_values("epoch")

        epochs_train = train_df["epoch"].values
        epochs_valid = valid_df["epoch"].values

        # ------------------------------------------------------------------ #
        # Figure setup
        # ------------------------------------------------------------------ #
        fig, ax = plt.subplots(figsize=(8, 6))

        TRAIN_COLOR = "black"
        VALID_FACE  = "white"
        VALID_EDGE  = "black"
        LINEWIDTH   = 2
        MARKERSIZE  = 6           # as requested (matplotlib uses points²-ish for `ms`)

        # ------------------------------------------------------------------ #
        # CASE 1 — single metric
        # ------------------------------------------------------------------ #
        if len(data_type) == 1:
            metric   = data_type[0]
            col      = metric                   # column name in df
            label    = LABEL_MAP.get(metric, metric.capitalize())
            marker   = MARKER_MAP.get(metric, "o")

            if metric.lower() == "dropout":
                if metric.lower() not in df.columns:
                    for col in df.columns:
                        if metric.lower() in col:
                            break
            y_train  = train_df[col].values
            y_valid  = valid_df[col].values

            ax.plot(epochs_train, y_train,
                    color=TRAIN_COLOR, linestyle="-", linewidth=LINEWIDTH,
                    marker=marker, markersize=MARKERSIZE,
                    label="train")

            if "dropout" not in metric:
                ax.plot(epochs_valid, y_valid,
                        color=TRAIN_COLOR, linestyle="-", linewidth=LINEWIDTH,
                        marker=marker, markersize=MARKERSIZE,
                        markerfacecolor=VALID_FACE, markeredgecolor=VALID_EDGE,
                        label="valid")

            
                # Legend position: loss → upper right, others → lower right
                ax.legend(loc=legend_loc, fontsize=14)

            # Y-axis title = metric name (single metric only)
            if "dropout" in metric:
                label = "Dropout"
            ax.set_ylabel(label, fontsize=20, fontweight="bold")

        # ------------------------------------------------------------------ #
        # CASE 2 — two metrics
        # ------------------------------------------------------------------ #
        else:
            metric_a, metric_b = data_type[0], data_type[1]
            label_a  = LABEL_MAP.get(metric_a, metric_a.capitalize())
            label_b  = LABEL_MAP.get(metric_b, metric_b.capitalize())
            marker_a = MARKER_MAP.get(metric_a, "o")
            marker_b = MARKER_MAP.get(metric_b, "^")

            # metric_a train
            ax.plot(epochs_train, train_df[metric_a].values,
                    color=TRAIN_COLOR, linestyle="-", linewidth=LINEWIDTH,
                    marker=marker_a, markersize=MARKERSIZE,
                    label=f"{label_a} (train)")

            # metric_a valid
            ax.plot(epochs_valid, valid_df[metric_a].values,
                    color=TRAIN_COLOR, linestyle="-", linewidth=LINEWIDTH,
                    marker=marker_a, markersize=MARKERSIZE,
                    markerfacecolor=VALID_FACE, markeredgecolor=VALID_EDGE,
                    label=f"{label_a} (valid)")

            # metric_b train
            ax.plot(epochs_train, train_df[metric_b].values,
                    color=TRAIN_COLOR, linestyle="-", linewidth=LINEWIDTH,
                    marker=marker_b, markersize=MARKERSIZE,
                    label=f"{label_b} (train)")

            # metric_b valid
            ax.plot(epochs_valid, valid_df[metric_b].values,
                    color=TRAIN_COLOR, linestyle="-", linewidth=LINEWIDTH,
                    marker=marker_b, markersize=MARKERSIZE,
                    markerfacecolor=VALID_FACE, markeredgecolor=VALID_EDGE,
                    label=f"{label_b} (valid)")

            # No y-axis title for two-metric case
            ax.set_ylabel("")

            # Legend: if either metric is loss → upper right, else lower right
            ax.legend(loc=legend_loc, fontsize=14)

        # ------------------------------------------------------------------ #
        # Shared axis formatting
        # ------------------------------------------------------------------ #
        ax.set_xlabel("Epoch", fontsize=20, fontweight="bold")

        ax.tick_params(axis="both", which="both",
                       direction="out",
                       labelsize=16)

        # Bold tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
            label.set_fontsize(16)

        # ------------------------------------------------------------------ #
        # Save
        # ------------------------------------------------------------------ #
        plt.tight_layout()

        fname = f"{out_folder}/LearnCurve_" + "_".join(data_type) + ".jpg"

        # 150 pixels per cm → convert to DPI (1 inch = 2.54 cm)
        dpi = 150 * 2.54                        # ≈ 381 DPI

        fig.savefig(fname, dpi=dpi, format="jpg", bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {fname}")

    except Exception:
        with open("plot_LearnCurve_error.txt", "w") as f:
            f.write(traceback.format_exc())
        raise
        
def plot_single_metric(train_epochs: List[int], valid_epochs: List[int],
                      train_values: List[float], valid_values: List[float],
                      metric_name: str, ylabel: str = None, 
                      save_path: str = None, show_plot: bool = False) -> plt.Figure:
    """
    Plot a single metric for training and validation.
    
    Args:
        train_epochs: List of training epoch numbers
        valid_epochs: List of validation epoch numbers
        train_values: Training metric values
        valid_values: Validation metric values
        metric_name: Name of the metric for title
        ylabel: Y-axis label (defaults to metric_name)
        save_path: Path to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training and validation curves
    ax.plot(train_epochs, train_values, 'b-', label=f'Train {metric_name}', linewidth=2, marker='o', markersize=3)
    ax.plot(valid_epochs, valid_values, 'r-', label=f'Valid {metric_name}', linewidth=2, marker='s', markersize=3)
    
    # Customize plot
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel or metric_name, fontsize=12)
    ax.set_title(f'{metric_name} Over Training Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    if train_values or valid_values:
        all_values = list(train_values) + list(valid_values)
        y_min, y_max = min(all_values), max(all_values)
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return fig

def plot_confusion_matrix_components(train_epochs: List[int], valid_epochs: List[int],
                                   train_data: Dict[str, List[int]], valid_data: Dict[str, List[int]],
                                   save_dir: str = None, show_plot: bool = False) -> plt.Figure:
    """
    Plot confusion matrix components (TP, TN, FP, FN) in a 2x2 subplot.
    
    Args:
        train_epochs: List of training epoch numbers
        valid_epochs: List of validation epoch numbers
        train_data: Dictionary with keys 'tp', 'tn', 'fp', 'fn' for training
        valid_data: Dictionary with keys 'tp', 'tn', 'fp', 'fn' for validation
        save_dir: Directory to save the plot
        show_plot: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    axes = [ax1, ax2, ax3, ax4]
    components = ['tp', 'tn', 'fp', 'fn']
    titles = ['True Positives (TP)', 'True Negatives (TN)', 'False Positives (FP)', 'False Negatives (FN)']
    colors = [('blue', 'red'), ('green', 'orange'), ('purple', 'brown'), ('pink', 'gray')]
    
    for ax, component, title, (train_color, valid_color) in zip(axes, components, titles, colors):
        train_vals = train_data.get(component, [])
        valid_vals = valid_data.get(component, [])
        
        ax.plot(train_epochs, train_vals, color=train_color, label=f'Train {component.upper()}', 
                linewidth=2, marker='o', markersize=3)
        ax.plot(valid_epochs, valid_vals, color=valid_color, label=f'Valid {component.upper()}', 
                linewidth=2, marker='s', markersize=3)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y limits
        if train_vals or valid_vals:
            all_vals = list(train_vals) + list(valid_vals)
            y_max = max(all_vals) if all_vals else 1
            ax.set_ylim(0, y_max * 1.1)
    
    fig.suptitle('Confusion Matrix Components Over Training', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot if directory provided
    if save_dir:
        save_path = os.path.join(save_dir, 'confusion_matrix_components.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return fig

def plot_all_metrics(train_epochs: List[int], valid_epochs: List[int],
                     train_data: Dict[str, List], valid_data: Dict[str, List],
                     epoch: int, save_dir: str = None, title_prefix: str = None) -> List[plt.Figure]:
    """
    Plot all metrics as individual graphs.
    
    Args:
        train_epochs: List of training epoch numbers
        valid_epochs: List of validation epoch numbers
        train_data: Training data dictionary with all metrics
        valid_data: Validation data dictionary with all metrics
        epoch: Current epoch for filename
        save_dir: Directory to save plots
        title_prefix: Prefix for plot titles
        
    Returns:
        List of matplotlib Figure objects
    """
    figures = []
    
    # Define metrics to plot individually
    metrics_config = {
        'loss': {'ylabel': 'Loss', 'filename': 'loss'},
        'acc': {'ylabel': 'Accuracy', 'filename': 'accuracy'},
        'f1': {'ylabel': 'F1 Score', 'filename': 'f1_score'},
        'sens': {'ylabel': 'Sensitivity (Recall)', 'filename': 'sensitivity'},
        'spec': {'ylabel': 'Specificity', 'filename': 'specificity'},
        'prec': {'ylabel': 'Precision', 'filename': 'precision'},
        'dropout': {'ylabel': 'Dropout Rate', 'filename': 'dropout'}
    }
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot each metric individually
    for metric, config in metrics_config.items():
        if metric in train_data and metric in valid_data:
            train_values = train_data[metric]
            valid_values = valid_data[metric]
            
            # Skip if no data
            if not train_values and not valid_values:
                continue
                
            # Create title
            plot_title = metric.upper()
            if title_prefix:
                plot_title = f"{title_prefix} - {plot_title}"
            
            # Create save path
            save_path = None
            if save_dir:
                filename = f"{config['filename']}_epoch_{epoch}.png"
                save_path = os.path.join(save_dir, filename)
            
            # Plot the metric
            fig = plot_single_metric(
                train_epochs=train_epochs,
                valid_epochs=valid_epochs,
                train_values=train_values,
                valid_values=valid_values,
                metric_name=plot_title,
                ylabel=config['ylabel'],
                save_path=save_path,
                show_plot=False
            )
            
            figures.append(fig)
    
    # Plot confusion matrix components
    confusion_data = {
        'train': {comp: train_data.get(comp, []) for comp in ['tp', 'tn', 'fp', 'fn']},
        'valid': {comp: valid_data.get(comp, []) for comp in ['tp', 'tn', 'fp', 'fn']}
    }
    
    fig = plot_confusion_matrix_components(
        train_epochs=train_epochs,
        valid_epochs=valid_epochs,
        train_data=confusion_data['train'],
        valid_data=confusion_data['valid'],
        save_dir=save_dir,
        show_plot=False
    )
    figures.append(fig)
    # plt.close(fig)
    return figures

def create_summary_plot(train_epochs: List[int], valid_epochs: List[int],
                       train_data: Dict[str, List], valid_data: Dict[str, List],
                       epoch: int, save_dir: str = None, title: str = None) -> plt.Figure:
    """
    Create a summary plot with key metrics in a multi-panel layout.
    
    Args:
        train_epochs: List of training epoch numbers
        valid_epochs: List of validation epoch numbers
        train_data: Training data dictionary
        valid_data: Validation data dictionary
        epoch: Current epoch
        save_dir: Directory to save the plot
        title: Overall title for the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Define key metrics for summary
    summary_metrics = [
        ('loss', 'Loss', 'Loss'),
        ('acc', 'Accuracy', 'Accuracy'),
        ('f1', 'F1 Score', 'F1 Score'),
        ('sens', 'Sensitivity', 'Sensitivity (Recall)'),
        ('spec', 'Specificity', 'Specificity'),
        ('prec', 'Precision', 'Precision')
    ]
    
    for i, (metric, plot_title, ylabel) in enumerate(summary_metrics):
        ax = axes[i]
        
        if metric in train_data and metric in valid_data:
            train_vals = train_data[metric]
            valid_vals = valid_data[metric]
            
            if train_vals or valid_vals:
                ax.plot(train_epochs, train_vals, 'b-', label=f'Train', linewidth=2, marker='o', markersize=2)
                ax.plot(valid_epochs, valid_vals, 'r-', label=f'Valid', linewidth=2, marker='s', markersize=2)
                
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel(ylabel, fontsize=10)
                ax.set_title(plot_title, fontsize=12, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{plot_title}\nNo Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    else:
        fig.suptitle(f'Training Summary - Epoch {epoch}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'training_summary_epoch_{epoch}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.close(fig)
    return fig

