## Pearson delta vs control bias
## Pearson delta vs n0
## Pearson delta affected genes vs # perturbations
# Figsize (7, 6)
# Fontsize 14 (xy labels), 16 (title)

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
import argparse

# Plot directory will be created in main function

# Set up a clean, professional style similar to science plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper")
sns.set_style("ticks")

PDS_LABELS = {
    'pds_l1': 'PDS (L1)',
    'pds_l2': 'PDS (L2)',
    'pds_cosine': 'PDS (Cosine)',
}

# Set matplotlib parameters to create professional plots
plt.rcParams.update({
    # Figure aesthetics
    'figure.facecolor': 'white',
    'figure.figsize': (7, 6),  # Updated to match requested figsize
    'figure.dpi': 300,
    
    # Text properties
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'sans-serif'],
    'font.size': 12,
    'axes.titlesize': 18.5,    # Updated to 18.5
    'axes.labelsize': 16,      # Updated to 16
    'xtick.labelsize': 14,     # Updated to 14
    'ytick.labelsize': 14,     # Updated to 14
    
    # Axes properties
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    
    # Tick properties
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    
    # Legend properties
    'legend.frameon': False,
    'legend.fontsize': 15,     # Updated to 15
    'legend.title_fontsize': 15,  # Updated to 15
    
    # Saving properties
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

def load_data(file_path):
    """Load dataset from the specified CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    return pd.read_csv(file_path)

def moving_average(x, y, window=10):
    """Calculate moving average of y with respect to sorted x values."""
    # Sort points by x values
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]
    
    # Calculate moving average
    x_ma = []
    y_ma = []
    
    # Use a sliding window to calculate moving averages
    for i in range(len(x_sorted) - window + 1):
        x_ma.append(np.mean(x_sorted[i:i+window]))
        y_ma.append(np.mean(y_sorted[i:i+window]))
    
    return np.array(x_ma), np.array(y_ma)

def resolve_pds_metric(data):
    """Resolve which PDS metric column to use, preferring cosine then l2 then l1."""
    for candidate in ('pds_cosine', 'pds_l2', 'pds_l1'):
        if candidate in data.columns:
            return candidate
    raise ValueError("No PDS metric columns found in results (expected pds_cosine, pds_l2, or pds_l1).")

def prepare_xy(x, y):
    """Convert x/y to numeric arrays and drop non-finite values."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]

def plot_metric_vs_parameter(data, save_path, x_column, y_column, title, x_label, y_label,
                             window=50, y_lim=None, log_x=False, log_x_if_range=False,
                             corr_log_x=None, log_x_range_threshold=10):
    """Generic scatter + moving-average plot with Pearson correlation annotation."""
    fig, ax = plt.subplots(figsize=(7, 6))

    y_series = pd.to_numeric(data[y_column], errors='coerce')
    x_series = data[x_column]
    x, y = prepare_xy(x_series, y_series)

    # Create scatter plot with default seaborn blue dots
    ax.scatter(x, y, alpha=0.3, s=20)

    # Decide whether to use log scale on x-axis
    use_log = log_x
    if log_x_if_range and len(x) > 0:
        min_x = np.min(x)
        max_x = np.max(x)
        if min_x > 0 and (max_x / min_x) > log_x_range_threshold:
            use_log = True

    # Calculate Pearson correlation
    if corr_log_x is None:
        corr_log_x = use_log
    if corr_log_x:
        corr_mask = x > 0
        x_corr = x[corr_mask]
        y_corr = y[corr_mask]
        corr_x = np.log10(x_corr) if len(x_corr) > 0 else x_corr
    else:
        x_corr = x
        y_corr = y
        corr_x = x_corr

    if len(x_corr) >= 2:
        corr, p_value = stats.pearsonr(corr_x, y_corr)
    else:
        corr, p_value = np.nan, np.nan

    # Add moving average trend line with specified window
    x_ma, y_ma = moving_average(np.array(x), np.array(y), window=window)
    if len(x_ma) > 0:
        ax.plot(x_ma, y_ma, color='navy', linestyle='--', linewidth=2)

    # Add title
    ax.set_title(title, fontsize=18.5, pad=20)

    # Set axis labels
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)

    if y_lim is not None:
        ax.set_ylim(*y_lim)

    # Add correlation text at leftmost limit with lowered y position
    ax.text(0.05, 0.97, f'Pearson R={corr:.2f}, P={p_value:.2e}',
            transform=ax.transAxes, fontsize=15, va='bottom', ha='left')

    # Use log scale for x-axis if requested
    if use_log:
        ax.set_xscale('log')

    # Remove top and right spines
    sns.despine()

    # Save figure as PDF
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Generated {save_path}")

def plot_pds_vs_parameter(data, save_path, x_column, title, x_label,
                          window=50, pds_metric=None, log_x=False, log_x_if_range=False):
    """Plot PDS vs a parameter with a moving average trend line."""
    if pds_metric is None:
        pds_metric = resolve_pds_metric(data)

    pds_label = PDS_LABELS.get(pds_metric, pds_metric)
    plot_metric_vs_parameter(
        data=data,
        save_path=save_path,
        x_column=x_column,
        y_column=pds_metric,
        title=title,
        x_label=x_label,
        y_label=f'Median {pds_label}',
        window=window,
        y_lim=(0.0, 1.1),
        log_x=log_x,
        log_x_if_range=log_x_if_range,
    )

def plot_pearson_delta_vs_parameter(data, save_path, x_column, title, x_label,
                                    window=50, y_column='pearson_all_median',
                                    y_label=r'Median Pearson$(\Delta^{p},\Delta^{all})$',
                                    log_x=False, log_x_if_range=False, corr_log_x=None):
    """Plot Pearson delta vs a parameter with a moving average trend line."""
    plot_metric_vs_parameter(
        data=data,
        save_path=save_path,
        x_column=x_column,
        y_column=y_column,
        title=title,
        x_label=x_label,
        y_label=y_label,
        window=window,
        y_lim=(-1.0, 1.1),
        log_x=log_x,
        log_x_if_range=log_x_if_range,
        corr_log_x=corr_log_x,
    )

def plot_sparsity(data, save_path):
    """
    Plot the histogram of sparsity (between 0.0 - 1.0)
    
    :param data: Description
    :param save_path: Description
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    sparsity = data["sparsity"]
    ax.hist(sparsity, bins=30, color='skyblue', edgecolor='black')
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Sparsity')
    
    # Save figure as PDF
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Generated {save_path}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate Pearson delta and PDS plots from simulation results.')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to the CSV file with simulation results')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Load data from the specified results file
        data = load_data(args.results)
        
        window = 100
        
        # Define save paths for all plots
        plot_dir = 'analyses/synthetic_simulations/paper_plots'
        os.makedirs(plot_dir, exist_ok=True)
        
        control_bias_path = os.path.join(plot_dir, 'pearson_delta_vs_control_bias.pdf')
        n0_path = os.path.join(plot_dir, 'pearson_delta_vs_n0.pdf')
        perturbations_path = os.path.join(plot_dir, 'pearson_delta_affected_vs_perturbations.pdf')
        pds_metric = resolve_pds_metric(data)
        pds_control_bias_path = os.path.join(plot_dir, f'{pds_metric}_vs_control_bias.pdf')
        pds_n0_path = os.path.join(plot_dir, f'{pds_metric}_vs_n0.pdf')
        pds_perturbations_path = os.path.join(plot_dir, f'{pds_metric}_vs_perturbations.pdf')
        sparsity_path = os.path.join(plot_dir, 'sparsity.pdf')
        mse_p_effect_path = os.path.join(plot_dir, 'mse_vs_p_effect.pdf')
        
        # Generate all plots
        # Plot Pearson delta vs control bias (β)
        plot_pearson_delta_vs_parameter(
            data=data,
            save_path=control_bias_path,
            x_column='B',
            title=r'$\mathbf{Pearson(Δ)}$ $\mathbf{by}$ $\mathbf{β}$  $\mathbf{(Simulation)}$',
            x_label='Control Bias (β)',
            window=window,
            corr_log_x=False,
        )
        # Plot Pearson delta vs n0
        plot_pearson_delta_vs_parameter(
            data=data,
            save_path=n0_path,
            x_column='N0',
            title=r'$\mathbf{Pearson(Δ)}$ $\mathbf{by}$ $\mathbf{n_0}$ $\mathbf{(Simulation)}$',
            x_label='Number of Control Cells ($n_0$)',
            window=window,
            log_x=True,
            corr_log_x=True,
        )
        # Plot Pearson delta (affected genes) vs number of perturbations.
        plot_pearson_delta_vs_parameter(
            data=data,
            save_path=perturbations_path,
            x_column='P',
            y_column='pearson_affected_median',
            title=r'$\mathbf{Pearson(Δ)}$ $\mathbf{by}$ $\mathbf{k}$ $\mathbf{(Simulation)}$',
            x_label='Number of Perturbations ($k$)',
            y_label=r'Median Pearson$(\Delta^{p},\Delta^{all})$ (Affected genes)',
            window=window,
            log_x_if_range=True,
            corr_log_x=True,
        )
        # Plot PDS vs control bias (β)
        plot_pds_vs_parameter(
            data=data,
            save_path=pds_control_bias_path,
            x_column='B',
            title=r'$\mathbf{PDS}$ $\mathbf{by}$ $\mathbf{β}$  $\mathbf{(Simulation)}$',
            x_label='Control Bias (β)',
            window=window,
            pds_metric=pds_metric,
            log_x=False,
        )
        # Plot PDS vs N0 (number of control cells)
        plot_pds_vs_parameter(
            data=data,
            save_path=pds_n0_path,
            x_column='N0',
            title=r'$\mathbf{PDS}$ $\mathbf{by}$ $\mathbf{n_0}$ $\mathbf{(Simulation)}$',
            x_label='Number of Control Cells ($n_0$)',
            window=window,
            pds_metric=pds_metric,
            log_x=True,
        )
        # Plot PDS vs number of perturbations
        plot_pds_vs_parameter(
            data=data,
            save_path=pds_perturbations_path,
            x_column='P',
            title=r'$\mathbf{PDS}$ $\mathbf{by}$ $\mathbf{k}$ $\mathbf{(Simulation)}$',
            x_label='Number of Perturbations ($k$)',
            window=window,
            pds_metric=pds_metric,
            log_x_if_range=True,
        )
        # Plot sparsity histogram
        plot_sparsity(data, sparsity_path)
        # Plot MSE vs p_effect (fraction of genes affected)
        plot_metric_vs_parameter(
            data=data,
            save_path=mse_p_effect_path,
            x_column='p_effect',
            y_column='mse_all_median',
            title=r'$\mathbf{MSE}$ $\mathbf{(Simulation)}$',
            x_label=r'Perturbation Probability ($\delta$)',
            y_label='MSE',
            window=window,
        )
        
        print(f"Successfully generated all plots from {args.results}")
    except Exception as e:
        print(f"Error generating plots: {e}")

if __name__ == "__main__":
    main()
