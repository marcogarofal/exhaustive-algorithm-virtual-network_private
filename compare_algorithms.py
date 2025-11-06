# compare_algorithms.py
"""
Script to compare exhaustive and greedy algorithms
Both algorithms receive the same graph configuration for fair comparison
"""

from tree_optimizer import run_algorithm as run_exhaustive
from greedy_algorithm_wrapper import run_greedy_algorithm
from graph_generator import generate_complete_config
from config_loader import load_config, create_default_config
import json
import os
import time


def compare_results(result_exhaustive, result_greedy, output_file='comparison_results.json'):
    """
    Compare results from exhaustive and greedy algorithms

    Args:
        result_exhaustive: result dict from exhaustive algorithm
        result_greedy: result dict from greedy algorithm
        output_file: file to save comparison
    """
    comparison = {
        'exhaustive_algorithm': {
            'execution_time': result_exhaustive['execution_time'],
            'num_nodes': result_exhaustive['num_nodes'],
            'num_edges': result_exhaustive['num_edges'],
            'best_tree_nodes': list(result_exhaustive['best_tree'].nodes()) if result_exhaustive['best_tree'] else [],
            'best_tree_edges': [list(edge) for edge in result_exhaustive['best_tree'].edges()] if result_exhaustive['best_tree'] else []
        },
        'greedy_algorithm': {
            'execution_time': result_greedy['execution_time'],
            'num_nodes': result_greedy['num_nodes'],
            'num_edges': result_greedy['num_edges'],
            'best_tree_nodes': list(result_greedy['best_tree'].nodes()) if result_greedy['best_tree'] else [],
            'best_tree_edges': [list(edge) for edge in result_greedy['best_tree'].edges()] if result_greedy['best_tree'] else [],
            'acc_cost': result_greedy.get('acc_cost', 0),
            'aoc_cost': result_greedy.get('aoc_cost', 0),
            'score': result_greedy.get('score', 0),
            'connected_weak': result_greedy.get('connected_weak', 0),
            'failed_connections': result_greedy.get('failed_connections', 0),
            'discretionary_used': result_greedy.get('discretionary_used', []),
            'alpha': result_greedy.get('alpha', 0.5)
        },
        'comparison': {
            'time_difference_seconds': result_greedy['execution_time'] - result_exhaustive['execution_time'],
            'speedup_factor': result_exhaustive['execution_time'] / result_greedy['execution_time'] if result_greedy['execution_time'] > 0 else float('inf'),
            'same_num_nodes': result_exhaustive['num_nodes'] == result_greedy['num_nodes'],
            'same_num_edges': result_exhaustive['num_edges'] == result_greedy['num_edges'],
            'same_solution': (set(result_exhaustive.get('best_tree_nodes', [])) == set(result_greedy.get('best_tree_nodes', [])) and
                            set(map(tuple, result_exhaustive.get('best_tree_edges', []))) == set(map(tuple, result_greedy.get('best_tree_edges', []))))
        }
    }

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    # Print detailed comparison
    print(f"\n{'='*80}")
    print("ALGORITHM COMPARISON RESULTS")
    print(f"{'='*80}")

    print(f"\n📊 EXHAUSTIVE ALGORITHM:")
    print(f"  ⏱️  Execution time: {result_exhaustive['execution_time']:.4f}s")
    print(f"  🔢 Nodes in solution: {result_exhaustive['num_nodes']}")
    print(f"  🔗 Edges in solution: {result_exhaustive['num_edges']}")

    print(f"\n📊 GREEDY ALGORITHM (α={result_greedy.get('alpha', 0.5)}):")
    print(f"  ⏱️  Execution time: {result_greedy['execution_time']:.4f}s")
    print(f"  🔢 Nodes in solution: {result_greedy['num_nodes']}")
    print(f"  🔗 Edges in solution: {result_greedy['num_edges']}")
    print(f"  ✅ Connected weak: {result_greedy.get('connected_weak', 0)}")
    print(f"  ❌ Failed connections: {result_greedy.get('failed_connections', 0)}")
    print(f"  🟠 Discretionary used: {result_greedy.get('discretionary_used', [])}")
    print(f"  📈 ACC (communication cost): {result_greedy.get('acc_cost', 0):.6f}")
    print(f"  📉 AOC (operational cost): {result_greedy.get('aoc_cost', 0):.6f}")
    print(f"  🎯 Total score: {result_greedy.get('score', 0):.2f}")

    print(f"\n🔍 COMPARISON:")
    speedup = comparison['comparison']['speedup_factor']
    if speedup == float('inf'):
        print(f"  ⚡ Speedup: INSTANT (greedy time ≈ 0)")
    else:
        print(f"  ⚡ Speedup: {speedup:.2f}x {'(greedy faster)' if speedup > 1 else '(exhaustive faster)'}")

    time_diff = comparison['comparison']['time_difference_seconds']
    print(f"  ⏱️  Time difference: {abs(time_diff):.4f}s {'(greedy saved)' if time_diff < 0 else '(exhaustive saved)'}")

    print(f"  🔢 Same number of nodes: {'✅ YES' if comparison['comparison']['same_num_nodes'] else '❌ NO'}")
    print(f"  🔗 Same number of edges: {'✅ YES' if comparison['comparison']['same_num_edges'] else '❌ NO'}")
    print(f"  🎯 Identical solution: {'✅ YES' if comparison['comparison']['same_solution'] else '❌ NO'}")

    print(f"\n💾 Detailed comparison saved to: {output_file}")
    print(f"{'='*80}\n")

    return comparison


if __name__ == "__main__":
    print("="*80)
    print("EXHAUSTIVE vs GREEDY ALGORITHM COMPARISON")
    print("="*80)

    # 1. Load or create configuration
    if not os.path.exists('config.json'):
        print("\n📝 Creating default config.json...")
        create_default_config('config.json')

    config = load_config('config.json')

    # 2. Generate graph configuration (same for both algorithms)
    print("\n🔧 Generating graph configuration...")
    graph_config = generate_complete_config(config)

    seed = config.get('graph_parameters', {}).get('seed', 42)
    alpha = config.get('algorithm', {}).get('alpha', 0.5)

    print(f"\n📊 Graph Configuration:")
    total_nodes = len(graph_config['weak_nodes']) + len(graph_config['power_nodes_mandatory']) + len(graph_config['power_nodes_discretionary'])
    print(f"  Total nodes: {total_nodes}")
    print(f"  Weak nodes: {len(graph_config['weak_nodes'])} → {graph_config['weak_nodes']}")
    print(f"  Mandatory nodes: {len(graph_config['power_nodes_mandatory'])} → {graph_config['power_nodes_mandatory']}")
    print(f"  Discretionary nodes: {len(graph_config['power_nodes_discretionary'])} → {graph_config['power_nodes_discretionary']}")
    print(f"  Random seed: {seed} (ensures same graph for both algorithms)")
    print(f"  Alpha (greedy): {alpha} (balance ACC/AOC)")

    # 3. Run EXHAUSTIVE algorithm
    print(f"\n{'='*80}")
    print("🔄 Running EXHAUSTIVE algorithm...")
    print(f"{'='*80}")

    algorithm_config_exhaustive = {'seed': seed}
    debug_config_exhaustive = {
        'plot_initial_graphs': False,
        'plot_intermediate': False,
        'plot_final': False,
        'save_plots': True,
        'verbose': False
    }

    result_exhaustive = run_exhaustive(
        graph_config=graph_config,
        algorithm_config=algorithm_config_exhaustive,
        debug_config=debug_config_exhaustive,
        output_dir='plots_exhaustive'
    )

    print(f"✅ Exhaustive algorithm completed in {result_exhaustive['execution_time']:.4f}s")
    print(f"   Solution: {result_exhaustive['num_nodes']} nodes, {result_exhaustive['num_edges']} edges")

    # 4. Run GREEDY algorithm (SAME GRAPH!)
    print(f"\n{'='*80}")
    print("🔄 Running GREEDY algorithm...")
    print(f"{'='*80}")

    algorithm_config_greedy = {
        'seed': seed,  # SAME SEED = SAME GRAPH!
        'alpha': alpha
    }

    result_greedy = run_greedy_algorithm(
        graph_config=graph_config,
        algorithm_config=algorithm_config_greedy,
        output_dir='plots_greedy'
    )

    print(f"✅ Greedy algorithm completed in {result_greedy['execution_time']:.4f}s")
    print(f"   Solution: {result_greedy['num_nodes']} nodes, {result_greedy['num_edges']} edges")
    print(f"   Score: {result_greedy.get('score', 0):.2f}")

    # 5. Compare results
    comparison = compare_results(result_exhaustive, result_greedy)

    # 6. Additional analysis
    print("\n📊 ADDITIONAL ANALYSIS:")

    # Check if exhaustive has scores.json
    scores_path = 'plots_exhaustive/scores.json'
    if os.path.exists(scores_path):
        with open(scores_path, 'r') as f:
            exhaustive_scores = json.load(f)

        print(f"\n  Exhaustive algorithm explored:")
        print(f"    - {len(exhaustive_scores['trees'])} total tree configurations")
        print(f"    - Global max weight: {exhaustive_scores['global_max_weight']}")
        print(f"    - Global max edges: {exhaustive_scores['global_max_edges']}")

        # Find best score in exhaustive
        best_exhaustive_score = min(tree_data['total'] for tree_data in exhaustive_scores['trees'].values())
        print(f"    - Best normalized score: {best_exhaustive_score:.4f}")

    print(f"\n  Greedy algorithm:")
    print(f"    - Tested only 2 configurations (with/without discretionary)")
    print(f"    - Much faster but potentially suboptimal")

    # Quality comparison
    if comparison['comparison']['same_solution']:
        print(f"\n  🎉 RESULT: Both algorithms found the SAME optimal solution!")
        print(f"     Greedy achieved optimality with {comparison['comparison']['speedup_factor']:.2f}x speedup!")
    else:
        print(f"\n  ⚠️  RESULT: Algorithms found DIFFERENT solutions")
        print(f"     Exhaustive is guaranteed optimal")
        print(f"     Greedy traded optimality for {comparison['comparison']['speedup_factor']:.2f}x speedup")

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}\n")
