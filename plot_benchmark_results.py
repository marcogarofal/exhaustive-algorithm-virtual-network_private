# plot_benchmark_results.py
"""
Plot benchmark results from raw data
Flexible plotting script to visualize algorithm comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os


# ======================== CONFIGURABLE PLOT SETTINGS ========================

# Which raw results file to load
RAW_RESULTS_FILE = 'benchmark_results/raw_results_20251109_200950.json'  # Change to your file

# Configurations to plot (define what you want to compare)
PLOT_CONFIGS = [
    # Example 1: Vary number of nodes (keep ratios, capacity, strategy fixed)
   # {'num_nodes': 5, 'weak_ratio': 0.6, 'mandatory_ratio': 0.2, 'capacity_config': 'default_10', 'weight_strategy': 'uniform'},
   # {'num_nodes': 6, 'weak_ratio': 0.2, 'mandatory_ratio': 0.2, 'capacity_config': 'default_10', 'weight_strategy': 'uniform'},
   {'num_nodes': 5, 'weak_ratio': 0.6, 'mandatory_ratio': 0.2, 'capacity_config': 'rand_1-5', 'weight_strategy': 'uniform'},
    {'num_nodes': 6, 'weak_ratio': 0.6, 'mandatory_ratio': 0.2, 'capacity_config': 'rand_1-5', 'weight_strategy': 'uniform'},

]

# What to plot
PLOT_TIMES = True          # Plot execution times
PLOT_MATCH_RATES = True    # Plot match rates
PLOT_SPEEDUP = True        # Plot speedup factors

# Output settings
OUTPUT_DIR = 'benchmark_plots'
FIGURE_SIZE = (12, 6)
DPI = 300

# =============================================================================


def load_raw_results(filepath):
    """Load raw results from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} test results from {filepath}")
    return data


def filter_results_by_config(all_results, target_config):
    """
    Filter results matching a specific configuration

    Args:
        all_results: list of all test results
        target_config: dict with keys to match (num_nodes, weak_ratio, etc.)

    Returns:
        list of matching test results
    """
    matching = []

    for test in all_results:
        match = True

        # Check each criterion
        if 'num_nodes' in target_config and test['config']['num_nodes'] != target_config['num_nodes']:
            match = False
        if 'weak_ratio' in target_config and test['config']['weak_ratio'] != target_config['weak_ratio']:
            match = False
        if 'mandatory_ratio' in target_config and test['config']['mandatory_ratio'] != target_config['mandatory_ratio']:
            match = False
        if 'capacity_config' in target_config and test['config']['capacity_config'] != target_config['capacity_config']:
            match = False
        if 'weight_strategy' in target_config and test['config']['weight_strategy'] != target_config['weight_strategy']:
            match = False

        if match:
            matching.append(test)

    return matching


def calculate_stats_for_config(filtered_results):
    """
    Calculate statistics for a set of filtered results

    Returns:
        dict with mean, std for each algorithm + match rates
    """
    stats = {
        'n_tests': len(filtered_results),
        'exhaustive_times': [],
        'greedy_times': [],
        'sa_times': [],
        'greedy_matches': 0,
        'sa_matches': 0
    }

    for test in filtered_results:
        # Collect times
        if 'error' not in test.get('exhaustive', {}):
            stats['exhaustive_times'].append(test['exhaustive']['time'])
        if 'error' not in test.get('greedy', {}):
            stats['greedy_times'].append(test['greedy']['time'])
            if test['greedy'].get('matches_exhaustive', False):
                stats['greedy_matches'] += 1
        if 'error' not in test.get('sa', {}):
            stats['sa_times'].append(test['sa']['time'])
            if test['sa'].get('matches_exhaustive', False):
                stats['sa_matches'] += 1

    # Calculate statistics
    result = {'n_tests': stats['n_tests']}

    for algo in ['exhaustive', 'greedy', 'sa']:
        times = stats[f'{algo}_times']
        if times:
            result[f'{algo}_mean'] = np.mean(times)
            result[f'{algo}_std'] = np.std(times)
            result[f'{algo}_median'] = np.median(times)
            result[f'{algo}_min'] = np.min(times)
            result[f'{algo}_max'] = np.max(times)
        else:
            result[f'{algo}_mean'] = 0
            result[f'{algo}_std'] = 0

    # Match rates
    result['greedy_match_rate'] = stats['greedy_matches'] / stats['n_tests'] if stats['n_tests'] > 0 else 0
    result['sa_match_rate'] = stats['sa_matches'] / stats['n_tests'] if stats['n_tests'] > 0 else 0
    result['greedy_match_count'] = stats['greedy_matches']
    result['sa_match_count'] = stats['sa_matches']

    # Speedup
    if result['exhaustive_mean'] > 0:
        result['greedy_speedup'] = result['exhaustive_mean'] / result['greedy_mean'] if result['greedy_mean'] > 0 else 0
        result['sa_speedup'] = result['exhaustive_mean'] / result['sa_mean'] if result['sa_mean'] > 0 else 0

    return result


def create_config_label(config):
    """Create a readable label for a configuration"""
    parts = []

    if 'num_nodes' in config:
        parts.append(f"{config['num_nodes']}n")
    if 'weak_ratio' in config:
        parts.append(f"w{int(config['weak_ratio']*100)}")
    if 'mandatory_ratio' in config:
        parts.append(f"m{int(config['mandatory_ratio']*100)}")
    if 'capacity_config' in config:
        parts.append(config['capacity_config'].replace('default_', 'c').replace('rand_', 'c'))
    if 'weight_strategy' in config:
        strat_short = config['weight_strategy'].replace('uniform', 'uni').replace('favor_discretionary', 'fav').replace('strong_favor', 'sfav')
        parts.append(strat_short)

    return '_'.join(parts)


def plot_execution_times(all_results, plot_configs, output_file='execution_times.png'):
    """
    Plot execution times comparison for different configurations

    Args:
        all_results: all raw test results
        plot_configs: list of configurations to plot
        output_file: output filename
    """
    # Calculate stats for each config
    config_stats = []
    config_labels = []

    for config in plot_configs:
        filtered = filter_results_by_config(all_results, config)
        stats = calculate_stats_for_config(filtered)
        config_stats.append(stats)
        config_labels.append(create_config_label(config))

        print(f"Config {create_config_label(config)}: {stats['n_tests']} tests, "
              f"Ex={stats['exhaustive_mean']:.3f}±{stats['exhaustive_std']:.3f}s")

    # Prepare data for plotting
    x = np.arange(len(config_labels))
    width = 0.25

    ex_means = [s['exhaustive_mean'] for s in config_stats]
    ex_stds = [s['exhaustive_std'] for s in config_stats]

    gr_means = [s['greedy_mean'] for s in config_stats]
    gr_stds = [s['greedy_std'] for s in config_stats]

    sa_means = [s['sa_mean'] for s in config_stats]
    sa_stds = [s['sa_std'] for s in config_stats]

    # Create plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    bars1 = ax.bar(x - width, ex_means, width, yerr=ex_stds,
                   label='Exhaustive', capsize=5, color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x, gr_means, width, yerr=gr_stds,
                   label='Greedy', capsize=5, color='#A23B72', alpha=0.8)
    bars3 = ax.bar(x + width, sa_means, width, yerr=sa_stds,
                   label='Simulated Annealing', capsize=5, color='#F18F01', alpha=0.8)

    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Execution Time Comparison\n(mean ± std over 100 seeds)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, output_file)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"✓ Execution times plot saved to {output_path}")


def plot_match_rates(all_results, plot_configs, output_file='match_rates.png'):
    """Plot match rates for different configurations"""

    config_stats = []
    config_labels = []

    for config in plot_configs:
        filtered = filter_results_by_config(all_results, config)
        stats = calculate_stats_for_config(filtered)
        config_stats.append(stats)
        config_labels.append(create_config_label(config))

    # Prepare data
    x = np.arange(len(config_labels))
    width = 0.35

    gr_rates = [s['greedy_match_rate'] * 100 for s in config_stats]
    sa_rates = [s['sa_match_rate'] * 100 for s in config_stats]

    # Create plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    bars1 = ax.bar(x - width/2, gr_rates, width, label='Greedy',
                   color='#A23B72', alpha=0.8)
    bars2 = ax.bar(x + width/2, sa_rates, width, label='Simulated Annealing',
                   color='#F18F01', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Match Rate with Exhaustive (%)', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Solution Quality\n(% of times matching exhaustive optimal solution)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.set_ylim([0, 110])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect (100%)')

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, output_file)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"✓ Match rates plot saved to {output_path}")


def plot_speedup(all_results, plot_configs, output_file='speedup.png'):
    """Plot speedup comparison"""

    config_stats = []
    config_labels = []

    for config in plot_configs:
        filtered = filter_results_by_config(all_results, config)
        stats = calculate_stats_for_config(filtered)
        config_stats.append(stats)
        config_labels.append(create_config_label(config))

    # Prepare data
    x = np.arange(len(config_labels))
    width = 0.35

    gr_speedup = [s.get('greedy_speedup', 0) for s in config_stats]
    sa_speedup = [s.get('sa_speedup', 0) for s in config_stats]

    # Create plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    bars1 = ax.bar(x - width/2, gr_speedup, width, label='Greedy',
                   color='#A23B72', alpha=0.8)
    bars2 = ax.bar(x + width/2, sa_speedup, width, label='Simulated Annealing',
                   color='#F18F01', alpha=0.8)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}x', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup vs Exhaustive (x times faster)', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Speedup Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Same speed as exhaustive')

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, output_file)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"✓ Speedup plot saved to {output_path}")


def plot_combined_view(all_results, plot_configs, output_file='combined_view.png'):
    """Create a combined view with multiple subplots"""

    config_stats = []
    config_labels = []

    for config in plot_configs:
        filtered = filter_results_by_config(all_results, config)
        stats = calculate_stats_for_config(filtered)
        config_stats.append(stats)
        config_labels.append(create_config_label(config))

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

    x = np.arange(len(config_labels))
    width = 0.25

    # Subplot 1: Execution times
    ex_means = [s['exhaustive_mean'] for s in config_stats]
    ex_stds = [s['exhaustive_std'] for s in config_stats]
    gr_means = [s['greedy_mean'] for s in config_stats]
    gr_stds = [s['greedy_std'] for s in config_stats]
    sa_means = [s['sa_mean'] for s in config_stats]
    sa_stds = [s['sa_std'] for s in config_stats]

    ax1.bar(x - width, ex_means, width, yerr=ex_stds, label='Exhaustive',
            capsize=5, color='#2E86AB', alpha=0.8)
    ax1.bar(x, gr_means, width, yerr=gr_stds, label='Greedy',
            capsize=5, color='#A23B72', alpha=0.8)
    ax1.bar(x + width, sa_means, width, yerr=sa_stds, label='SA',
            capsize=5, color='#F18F01', alpha=0.8)

    ax1.set_ylabel('Time (s)', fontweight='bold')
    ax1.set_title('Execution Times (mean ± std)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Subplot 2: Match rates
    gr_rates = [s['greedy_match_rate'] * 100 for s in config_stats]
    sa_rates = [s['sa_match_rate'] * 100 for s in config_stats]

    ax2.bar(x - width/2, gr_rates, width, label='Greedy', color='#A23B72', alpha=0.8)
    ax2.bar(x + width/2, sa_rates, width, label='SA', color='#F18F01', alpha=0.8)
    ax2.axhline(y=100, color='green', linestyle='--', alpha=0.5)

    ax2.set_ylabel('Match Rate (%)', fontweight='bold')
    ax2.set_title('Solution Quality (% matching exhaustive optimum)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_labels, rotation=45, ha='right')
    ax2.set_ylim([0, 110])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Subplot 3: Speedup
    gr_speedup = [s.get('greedy_speedup', 0) for s in config_stats]
    sa_speedup = [s.get('sa_speedup', 0) for s in config_stats]

    ax3.bar(x - width/2, gr_speedup, width, label='Greedy', color='#A23B72', alpha=0.8)
    ax3.bar(x + width/2, sa_speedup, width, label='SA', color='#F18F01', alpha=0.8)
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Configuration', fontweight='bold')
    ax3.set_ylabel('Speedup (x)', fontweight='bold')
    ax3.set_title('Speedup vs Exhaustive', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(config_labels, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, output_file)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()

    print(f"✓ Combined view saved to {output_path}")


def print_statistics_table(all_results, plot_configs):
    """Print a formatted table with statistics"""

    print("\n" + "="*120)
    print("STATISTICS TABLE FOR SELECTED CONFIGURATIONS")
    print("="*120)
    print(f"\n{'Config':<25} {'N':<6} {'Greedy Match':<18} {'SA Match':<18} "
          f"{'Ex Time':<20} {'Gr Time':<20} {'SA Time':<20}")
    print("-"*120)

    for config in plot_configs:
        filtered = filter_results_by_config(all_results, config)
        stats = calculate_stats_for_config(filtered)
        label = create_config_label(config)

        gr_match_str = f"{stats['greedy_match_count']}/{stats['n_tests']} ({stats['greedy_match_rate']*100:.1f}%)"
        sa_match_str = f"{stats['sa_match_count']}/{stats['n_tests']} ({stats['sa_match_rate']*100:.1f}%)"
        ex_time_str = f"{stats['exhaustive_mean']:.3f}±{stats['exhaustive_std']:.3f}s"
        gr_time_str = f"{stats['greedy_mean']:.4f}±{stats['greedy_std']:.4f}s"
        sa_time_str = f"{stats['sa_mean']:.3f}±{stats['sa_std']:.3f}s"

        print(f"{label:<25} {stats['n_tests']:<6} {gr_match_str:<18} {sa_match_str:<18} "
              f"{ex_time_str:<20} {gr_time_str:<20} {sa_time_str:<20}")

    print("="*120 + "\n")


def main():
    """Main plotting function"""

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("="*100)
    print("BENCHMARK RESULTS PLOTTER")
    print("="*100)

    # Load raw results
    print(f"\nLoading raw results from: {RAW_RESULTS_FILE}")
    all_results = load_raw_results(RAW_RESULTS_FILE)



    # DEBUG: Mostra le prime 3 configurazioni presenti nei dati
    print("\nDEBUG: First 3 test configurations in raw data:")
    for i, test in enumerate(all_results[:3]):
        print(f"  Test {i+1}: {test['config']}")



    print(f"\nConfigurations to plot: {len(PLOT_CONFIGS)}")
    for i, config in enumerate(PLOT_CONFIGS, 1):
        print(f"  {i}. {create_config_label(config)}")

    # Print statistics table
    print_statistics_table(all_results, PLOT_CONFIGS)

    # Generate plots
    print("\nGenerating plots...")

    if PLOT_TIMES:
        plot_execution_times(all_results, PLOT_CONFIGS)

    if PLOT_MATCH_RATES:
        plot_match_rates(all_results, PLOT_CONFIGS)

    if PLOT_SPEEDUP:
        plot_speedup(all_results, PLOT_CONFIGS)

    # Combined view
    plot_combined_view(all_results, PLOT_CONFIGS)

    print("\n✅ All plots generated successfully!")
    print(f"   Check {OUTPUT_DIR}/ directory for output files\n")


if __name__ == "__main__":
    main()
