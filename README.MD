# Tree Optimization Algorithm

A refactored and modular implementation of a tree optimization algorithm for graphs with different node types (weak, mandatory power, and discretionary power nodes).

## Features

- **Standalone Mode**: Run directly from command line with configuration files
- **Library Mode**: Import and use in your own Python scripts
- **Deterministic**: Support for random seeds for reproducible results
- **Configurable**: JSON-based configuration for easy parameter tuning
- **Algorithm Comparison Ready**: Designed to facilitate comparison with other algorithms
- **Multiple Algorithms**: Includes both exhaustive and greedy algorithms for comparison

## File Structure

```
project/
├── tree_optimizer.py              # Exhaustive algorithm implementation
├── greedy_steiner.py             # Greedy algorithm core functions
├── greedy_algorithm_wrapper.py   # Wrapper for greedy algorithm
├── compare_algorithms.py         # Compare exhaustive vs greedy
├── graph_generator.py            # Graph configuration generation
├── config_loader.py              # Configuration file management
├── config.json                   # Configuration file
└── README.md                     # This file
```

## Installation

### Requirements

```bash
pip install networkx matplotlib
```

### Quick Start

1. Create a default configuration file:
```bash
python config_loader.py create config.json
```

2. Run the exhaustive algorithm:
```bash
python tree_optimizer.py --config config.json
```

3. Compare exhaustive and greedy algorithms:
```bash
python compare_algorithms.py
```

## Usage

### 1. Standalone Mode - Exhaustive Algorithm

Run the exhaustive algorithm directly with a configuration file:

```bash
python tree_optimizer.py --config config.json
```

Or run with default hardcoded configuration:

```bash
python tree_optimizer.py
```

### 2. Algorithm Comparison Mode

Compare exhaustive and greedy algorithms on the same graph:

```bash
python compare_algorithms.py
```

This will:
- Load configuration from `config.json`
- Generate the same graph for both algorithms (using same seed)
- Run exhaustive algorithm → saves to `plots_exhaustive/`
- Run greedy algorithm → saves to `plots_greedy/`
- Compare results and save to `comparison_results.json`
- Display detailed comparison metrics

**Output includes**:
- Execution time comparison
- Solution quality comparison
- Speedup factor
- Whether both algorithms found the same solution
- Detailed metrics (nodes, edges, costs)

### 3. Library Mode

Use the algorithms as libraries in your own code:

#### Exhaustive Algorithm
```python
from tree_optimizer import run_algorithm
from graph_generator import generate_complete_config
from config_loader import load_config

# Load configuration
config = load_config('config.json')
graph_config = generate_complete_config(config)

# Run exhaustive algorithm
result = run_algorithm(
    graph_config=graph_config,
    algorithm_config=config.get('algorithm', {}),
    debug_config=config.get('debug', {}),
    output_dir='plots'
)

print(f"Execution time: {result['execution_time']:.2f} seconds")
print(f"Best tree has {result['num_nodes']} nodes and {result['num_edges']} edges")
```

#### Greedy Algorithm
```python
from greedy_algorithm_wrapper import run_greedy_algorithm
from graph_generator import generate_complete_config
from config_loader import load_config

# Load configuration
config = load_config('config.json')
graph_config = generate_complete_config(config)

# Run greedy algorithm
algorithm_config = {
    'seed': 42,
    'alpha': 0.5  # Balance between communication and operational costs
}

result = run_greedy_algorithm(
    graph_config=graph_config,
    algorithm_config=algorithm_config
)

print(f"Execution time: {result['execution_time']:.4f}s")
print(f"Score: {result['score']:.2f}")
print(f"ACC: {result['acc_cost']:.6f}, AOC: {result['aoc_cost']:.6f}")
```

#### Compare Both Algorithms Programmatically
```python
from tree_optimizer import run_algorithm as run_exhaustive
from greedy_algorithm_wrapper import run_greedy_algorithm
from graph_generator import generate_complete_config
from config_loader import load_config

# Load and generate same graph configuration
config = load_config('config.json')
graph_config = generate_complete_config(config)
seed = 42

# Run exhaustive
result_exhaustive = run_exhaustive(
    graph_config=graph_config,
    algorithm_config={'seed': seed},
    debug_config={'save_plots': True},
    output_dir='plots_exhaustive'
)

# Run greedy with SAME graph (same seed)
result_greedy = run_greedy_algorithm(
    graph_config=graph_config,
    algorithm_config={'seed': seed, 'alpha': 0.5}
)

# Compare
print(f"Exhaustive: {result_exhaustive['execution_time']:.4f}s")
print(f"Greedy: {result_greedy['execution_time']:.4f}s")
print(f"Speedup: {result_exhaustive['execution_time'] / result_greedy['execution_time']:.2f}x")
```

### 4. Programmatic Configuration

You can also define configuration programmatically without a JSON file:

```python
from tree_optimizer import run_algorithm

graph_config = {
    'weak_nodes': [1, 2, 3],
    'power_nodes_mandatory': [4, 5],
    'power_nodes_discretionary': [6, 7, 8],
    'capacities': {1: 10, 2: 30, 3: 2, 4: 1, 5: 10, 6: 4, 7: 5, 8: 5}
}

algorithm_config = {'seed': 42}
debug_config = {'plot_final': True, 'save_plots': True}

result = run_algorithm(graph_config, algorithm_config, debug_config)
```

## Configuration File

The `config.json` file controls all algorithm parameters:

### Graph Parameters

Define the graph structure using either **ratios** or **absolute numbers**:

**Option 1: Using ratios**
```json
{
  "graph_parameters": {
    "num_nodes": 8,
    "weak_ratio": 0.4,
    "mandatory_ratio": 0.2,
    "seed": 42
  }
}
```

**Option 2: Using absolute numbers**
```json
{
  "graph_parameters": {
    "num_weak": 3,
    "num_mandatory": 2,
    "num_discretionary": 3,
    "seed": 42
  }
}
```

### Node Capacities

Define capacity for each node with **multiple flexible options**:

#### Option 1: Default capacity for all nodes
```json
{
  "capacities": {
    "default": 10
  }
}
```
All nodes will have capacity 10.

#### Option 2: Default + Custom overrides (Recommended)
```json
{
  "capacities": {
    "default": 10,
    "custom": {
      "2": 30,
      "3": 2,
      "4": 1
    }
  }
}
```
Most nodes have capacity 10, but nodes 2, 3, and 4 have custom values. **Best for large graphs!**

#### Option 3: Random capacities
```json
{
  "capacities": {
    "random": {
      "min": 5,
      "max": 20,
      "seed": 42
    }
  }
}
```
Each node gets a random capacity between 5 and 20. Use `seed` for reproducibility.

#### Option 4: Random + Custom overrides
```json
{
  "capacities": {
    "random": {
      "min": 5,
      "max": 20,
      "seed": 42
    },
    "custom": {
      "2": 100,
      "3": 1
    }
  }
}
```
Random capacities for most nodes, but nodes 2 and 3 have fixed values.

#### Option 5: Explicit (Old format - for small graphs only)
```json
{
  "capacities": {
    "1": 10, "2": 30, "3": 2, "4": 1,
    "5": 10, "6": 4, "7": 5, "8": 5
  }
}
```
Specify each node individually. **Not recommended for graphs with many nodes.**

### Debug Options

Control visualization and logging:

```json
{
  "debug": {
    "plot_initial_graphs": false,
    "plot_intermediate": false,
    "plot_final": true,
    "save_plots": true,
    "verbose": false,
    "verbose_level2": false,
    "verbose_level3": false
  }
}
```

- `plot_initial_graphs`: Show initial graph visualizations
- `plot_intermediate`: Show intermediate step visualizations
- `plot_final`: Show final result visualization
- `save_plots`: Save plots to disk
- `verbose`: Enable detailed logging (level 1)
- `verbose_level2`: Enable more detailed logging (level 2)
- `verbose_level3`: Enable most detailed logging (level 3)

### Algorithm Configuration

```json
{
  "algorithm": {
    "weight_range": [1, 10],
    "alpha": 0.5
  }
}
```

- `weight_range`: Range for edge weights (exhaustive algorithm)
- `alpha`: Balance parameter for greedy algorithm (0.0 to 1.0)
  - `alpha = 1.0`: Only communication cost (ACC)
  - `alpha = 0.0`: Only operational cost (AOC)
  - `alpha = 0.5`: Balanced (default)

### Output Configuration

```json
{
  "output": {
    "plots_dir": "plots",
    "results_dir": "results"
  }
}
```

## Configuration Management

### Create Default Configuration

```bash
python config_loader.py create config.json
```

### Validate Configuration

```bash
python config_loader.py validate config.json
```

### Show Configuration

```bash
python config_loader.py show config.json
```

## Algorithm Comparison Workflow

### Quick Comparison

The simplest way to compare algorithms:

```bash
# 1. Create or edit config.json with your parameters
python config_loader.py create config.json

# 2. Run comparison
python compare_algorithms.py
```

This will generate:
- `plots_exhaustive/` - Images and scores.json from exhaustive algorithm
- `plots_greedy/` - (currently no plots from greedy)
- `comparison_results.json` - Detailed comparison metrics

### Understanding Comparison Results

The `comparison_results.json` file contains:

```json
{
  "exhaustive_algorithm": {
    "execution_time": 2.5432,
    "num_nodes": 6,
    "num_edges": 5,
    "best_tree_nodes": [1, 2, 3, 4, 5, 6],
    "best_tree_edges": [[1,4], [2,4], [3,5], [4,5], [5,6]]
  },
  "greedy_algorithm": {
    "execution_time": 0.0234,
    "num_nodes": 6,
    "num_edges": 5,
    "acc_cost": 0.142857,
    "aoc_cost": 0.033333,
    "score": 176.19,
    "connected_weak": 3,
    "failed_connections": 0,
    "discretionary_used": [5, 6]
  },
  "comparison": {
    "time_difference_seconds": -2.5198,
    "speedup_factor": 108.72,
    "same_num_nodes": true,
    "same_num_edges": true,
    "same_solution": true
  }
}
```

**Key Metrics**:
- `speedup_factor`: How many times faster is greedy vs exhaustive
- `same_solution`: Whether both found identical trees
- `acc_cost` (greedy): Average Communication Cost
- `aoc_cost` (greedy): Average Operational Cost (capacity usage)

### Advanced: Custom Comparison Script

For more control, create your own comparison script:

```python
# my_comparison.py
from tree_optimizer import run_algorithm as run_exhaustive
from greedy_algorithm_wrapper import run_greedy_algorithm
from graph_generator import generate_complete_config
from config_loader import load_config
import json

# Custom configuration
config = load_config('config.json')
graph_config = generate_complete_config(config)

# Test multiple alpha values for greedy
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
results = {}

# Run exhaustive once
result_exhaustive = run_exhaustive(
    graph_config=graph_config,
    algorithm_config={'seed': 42},
    debug_config={'save_plots': False}
)

# Run greedy with different alpha values
for alpha in alphas:
    result_greedy = run_greedy_algorithm(
        graph_config=graph_config,
        algorithm_config={'seed': 42, 'alpha': alpha}
    )

    results[f'alpha_{alpha}'] = {
        'greedy_time': result_greedy['execution_time'],
        'greedy_score': result_greedy['score'],
        'same_as_exhaustive': (
            set(result_exhaustive['best_tree'].nodes()) ==
            set(result_greedy['best_tree'].nodes())
        )
    }

# Save results
with open('alpha_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Alpha comparison complete!")
```

## Output Files

### Exhaustive Algorithm Outputs

When running exhaustive algorithm with `save_plots: true`:

**Directory**: `plots/` or `plots_exhaustive/`

- `0_graph.png` - Initial complete graph
- `1_first_phase.png`, `2_first_phase.png`, ... - Intermediate trees (without scores)
- `N_intermediate.png` - More intermediate trees
- `M_best_tree_e0.1714_d0.0667_t0.2381.png` - **Final best tree** with scores in filename and image
  - `e`: edge cost
  - `d`: degree cost
  - `t`: total cost
- `scores.json` - **All trees with globally normalized scores**

### scores.json Format

```json
{
  "global_max_weight": 5,
  "global_max_edges": 7,
  "trees": {
    "1_first_phase.png": {
      "edge_cost": 0.1714,
      "degree_cost": 0.0667,
      "total": 0.2381,
      "num_nodes": 6,
      "num_edges": 5,
      "nodes": [1, 2, 3, 4, 5, 6],
      "edges": [[1,4], [2,4], [3,5], [4,5], [5,6]]
    },
    "15_best_tree_e0.1714_d0.0667_t0.2381.png": {
      "edge_cost": 0.1714,
      "degree_cost": 0.0667,
      "total": 0.2381,
      "is_best": true
    }
  }
}
```

**All scores are comparable** because they use the same global normalization (max weight and max edges across ALL trees).

### Greedy Algorithm Outputs

Currently greedy algorithm doesn't save plots, only returns results in memory.

### Comparison Outputs

- `comparison_results.json` - Detailed comparison of both algorithms
- Console output with formatted comparison table

## Node Types

- **Weak Nodes** (green): Regular nodes with limited capabilities
- **Mandatory Power Nodes** (red): Must be included, have higher capacity
- **Discretionary Power Nodes** (orange): Optional high-capacity nodes

## Algorithm Behavior

### Exhaustive Algorithm
1. Creates initial graph with weak and mandatory nodes
2. Generates all valid tree combinations
3. Filters based on connectivity and hub constraints
4. Evaluates discretionary node combinations
5. Compares trees using edge costs and node degree costs
6. Returns the optimal tree (guaranteed optimal solution)

**Characteristics**:
- ✅ Guaranteed optimal solution
- ✅ Explores all possible combinations
- ❌ Slow for large graphs (exponential complexity)
- ✅ Provides scores for all evaluated trees

### Greedy Algorithm
1. Creates graph and connects mandatory nodes first
2. Finds all possible paths from weak to mandatory nodes
3. Greedily selects paths based on custom cost function
4. Tests only 2 configurations: without and with all discretionary nodes
5. Returns best solution found

**Characteristics**:
- ✅ Very fast (linear/polynomial complexity)
- ✅ Uses custom cost function: C(G) = α × ACC + (1-α) × AOC
  - ACC: Average Communication Cost (edge weights)
  - AOC: Average Operational Cost (capacity usage)
- ❌ Not guaranteed optimal (heuristic)
- ✅ Good solution quality in practice

## Reproducibility

For deterministic results, always set a `seed` in the configuration:

```json
{
  "graph_parameters": {
    "seed": 42
  }
}
```

This ensures:
- Same random edge weights
- Same graph generation
- Reproducible comparisons between algorithms
- Both algorithms receive identical graphs when compared

## Examples

### Example 1: Run Exhaustive Algorithm
```bash
python tree_optimizer.py --config config.json
```

### Example 2: Test Greedy Algorithm Wrapper
```bash
python greedy_algorithm_wrapper.py
```

### Example 3: Compare Both Algorithms
```bash
python compare_algorithms.py
```

### Example 4: Validate Configuration
```bash
python config_loader.py validate config.json
```

### Example 5: Custom Graph Size
Edit `config.json`:
```json
{
  "graph_parameters": {
    "num_nodes": 20,
    "weak_ratio": 0.3,
    "mandatory_ratio": 0.2,
    "seed": 42
  }
}
```
Then run comparison:
```bash
python compare_algorithms.py
```

## Benchmarking

### Running Comprehensive Benchmarks

The `benchmark.py` script performs full factorial testing across multiple dimensions:

```bash
python benchmark.py
```

**Test Matrix** (configurable at top of benchmark.py):
- **Number of nodes**: 10 different values (5-15 nodes)
- **Node ratios**: 3 combinations (weak/mandatory/discretionary percentages)
- **Capacities**: 3 configurations (low/medium/high)
- **Edge weight strategies**: 3 strategies (uniform, favor discretionary, strong favor)
- **Seeds**: 100 different random seeds per configuration

**Total**: 10 × 3 × 3 × 3 × 100 = **27,000 tests**

### Configuring Benchmark Parameters

Edit the top of `benchmark.py`:

```python
# Quick test (reduce for faster execution)
SEEDS_PER_CONFIG = 10              # Instead of 100
NUM_NODES_CONFIGS = [5, 8, 10]    # Instead of 10 values

# Full benchmark (publication-quality statistics)
SEEDS_PER_CONFIG = 100
NUM_NODES_CONFIGS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
```

### Benchmark Output Files

After running, find results in `benchmark_results/`:

1. **raw_results_TIMESTAMP.json** - All individual test results (27,000 entries)
   ```json
   [
     {
       "test_id": 1,
       "config": {"num_nodes": 5, "weak_ratio": 0.6, ...},
       "seed": 1,
       "exhaustive": {"time": 0.234, "num_nodes": 4, ...},
       "greedy": {"time": 0.012, "matches_exhaustive": true, ...},
       "sa": {"time": 0.089, "matches_exhaustive": false, ...}
     },
     ...
   ]
   ```

2. **aggregated_results_TIMESTAMP.json** - Statistics by dimension
   ```json
   {
     "by_num_nodes": {
       "5": {
         "n_tests": 900,
         "greedy_match_rate": 0.95,
         "exhaustive_time_mean": 0.234,
         "exhaustive_time_std": 0.015,
         ...
       }
     },
     "by_ratios": {...},
     "by_capacities": {...},
     "by_weight_strategy": {...}
   }
   ```

3. **benchmark_report_TIMESTAMP.txt** - Human-readable summary tables

### Understanding Benchmark Results

**Match Rate**: Percentage of times heuristic algorithms find the same solution as exhaustive (optimal)
- 100% = Always finds optimal solution
- 80% = Finds optimal in 80/100 cases
- Lower = More suboptimal solutions

**Time Statistics**:
- Mean ± Std over 100 seeds with same configuration
- Shows algorithm stability (low std = consistent performance)

**Speedup**: How many times faster than exhaustive
- 100x = Algorithm is 100 times faster
- Higher = Better for large-scale problems

## Plotting Benchmark Results

### Generate Plots from Benchmark Data

Use `plot_benchmark_results.py` to visualize results:

```bash
python plot_benchmark_results.py
```

### Configuring Plots

Edit `plot_benchmark_results.py` to select which configurations to plot:

```python
# Example 1: Compare different numbers of nodes (keep other params fixed)
PLOT_CONFIGS = [
    {'num_nodes': 5, 'weak_ratio': 0.4, 'mandatory_ratio': 0.2,
     'capacity_config': 'default_10', 'weight_strategy': 'uniform'},
    {'num_nodes': 8, 'weak_ratio': 0.4, 'mandatory_ratio': 0.2,
     'capacity_config': 'default_10', 'weight_strategy': 'uniform'},
    {'num_nodes': 10, 'weak_ratio': 0.4, 'mandatory_ratio': 0.2,
     'capacity_config': 'default_10', 'weight_strategy': 'uniform'},
]

# Example 2: Compare different weight strategies (keep size fixed)
PLOT_CONFIGS = [
    {'num_nodes': 8, 'weak_ratio': 0.4, 'mandatory_ratio': 0.2,
     'capacity_config': 'default_10', 'weight_strategy': 'uniform'},
    {'num_nodes': 8, 'weak_ratio': 0.4, 'mandatory_ratio': 0.2,
     'capacity_config': 'default_10', 'weight_strategy': 'favor_discretionary'},
    {'num_nodes': 8, 'weak_ratio': 0.4, 'mandatory_ratio': 0.2,
     'capacity_config': 'default_10', 'weight_strategy': 'strong_favor'},
]
```

### Finding Available Configurations

To see what configurations are in your benchmark results, add this helper to `plot_benchmark_results.py`:

```python
def list_available_configs(all_results):
    """List all unique configurations in the dataset"""
    unique_configs = set()
    for test in all_results:
        config_tuple = (
            test['config']['num_nodes'],
            test['config']['weak_ratio'],
            test['config']['mandatory_ratio'],
            test['config']['capacity_config'],
            test['config']['weight_strategy']
        )
        unique_configs.add(config_tuple)

    for cfg in sorted(unique_configs):
        print(f"nodes={cfg[0]}, weak={cfg[1]}, mand={cfg[2]}, cap={cfg[3]}, strat={cfg[4]}")
```

### Generated Plots

The script generates 4 plots in `benchmark_plots/`:

1. **execution_times.png** - Bar chart with error bars showing mean ± std execution times
2. **match_rates.png** - Bar chart showing % of times heuristics match exhaustive optimum
3. **speedup.png** - Bar chart showing speedup factors vs exhaustive
4. **combined_view.png** - All three plots combined in a single figure (3 subplots)

### Plot Customization

Configure plot appearance in `plot_benchmark_results.py`:

```python
PLOT_TIMES = True          # Plot execution times
PLOT_MATCH_RATES = True    # Plot match rates
PLOT_SPEEDUP = True        # Plot speedup factors
FIGURE_SIZE = (12, 6)      # Figure dimensions
DPI = 300                  # Resolution for publication
```

## Performance Notes

### "The graph is not fully connected"
The initial configuration produces a disconnected graph. Check node counts and connectivity requirements.

### "Missing required section"
Configuration file is incomplete. Use `python config_loader.py validate config.json` to check.

### Import errors
Ensure all required packages are installed: `pip install networkx matplotlib`

### "Module 'greedy_steiner' not found"
Make sure `greedy_steiner.py` is in the same directory as other scripts.

### Greedy algorithm errors
Ensure `greedy_steiner.py` and `greedy_algorithm_wrapper.py` are present and properly configured.

## Performance Notes

### Exhaustive Algorithm
- Complexity depends on number of nodes and discretionary nodes
- Use `yield` generators to avoid memory saturation with large graphs
- For large graphs, consider disabling visualization (`plot_intermediate: false`)
- Typical runtime: seconds to minutes depending on graph size

### Greedy Algorithm
- Much faster than exhaustive (typically 10-100x speedup)
- Suitable for large graphs where exhaustive is impractical
- Runtime: typically under 1 second for small to medium graphs
- Quality: often finds optimal or near-optimal solutions

### Comparison Recommendations
- Small graphs (< 10 nodes): Both algorithms feasible
- Medium graphs (10-15 nodes): Exhaustive may be slow but doable
- Large graphs (> 15 nodes): Use greedy, exhaustive may be impractical

## License

[Your license here]

## Contributing

[Contributing guidelines here]

## Contact

[Your contact information here]

## Installation

### Requirements

```bash
pip install networkx matplotlib
```

### Quick Start

1. Create a default configuration file:
```bash
python config_loader.py create config.json
```

2. Run a single algorithm:
```bash
python tree_optimizer.py --config config.json
```

3. Compare all three algorithms:
```bash
python compare_algorithms.py
```

4. Run comprehensive benchmark:
```bash
python benchmark.py
```

5. Generate plots from benchmark results:
```bash
python plot_benchmark_results.py
```

## Usage

### Standalone Mode

Run the algorithm directly with a configuration file:

```bash
python tree_optimizer.py --config config.json
```

Or run with default hardcoded configuration:

```bash
python tree_optimizer.py
```

### Library Mode

Use the algorithm as a library in your own code:

```python
from tree_optimizer import run_algorithm
from graph_generator import generate_complete_config
from config_loader import load_config

# Load configuration
config = load_config('config.json')
graph_config = generate_complete_config(config)

# Run algorithm
result = run_algorithm(
    graph_config=graph_config,
    algorithm_config=config.get('algorithm', {}),
    debug_config=config.get('debug', {}),
    output_dir='plots'
)

print(f"Execution time: {result['execution_time']:.2f} seconds")
print(f"Best tree has {result['num_nodes']} nodes and {result['num_edges']} edges")
```

### Programmatic Configuration

You can also define configuration programmatically without a JSON file:

```python
from tree_optimizer import run_algorithm

graph_config = {
    'weak_nodes': [1, 2, 3],
    'power_nodes_mandatory': [4, 5],
    'power_nodes_discretionary': [6, 7, 8],
    'capacities': {1: 10, 2: 30, 3: 2, 4: 1, 5: 10, 6: 4, 7: 5, 8: 5}
}

algorithm_config = {'seed': 42}
debug_config = {'plot_final': True, 'save_plots': True}

result = run_algorithm(graph_config, algorithm_config, debug_config)
```

## Configuration File

The `config.json` file controls all algorithm parameters:

### Graph Parameters

Define the graph structure using either **ratios** or **absolute numbers**:

**Option 1: Using ratios**
```json
{
  "graph_parameters": {
    "num_nodes": 8,
    "weak_ratio": 0.4,
    "mandatory_ratio": 0.2,
    "seed": 42
  }
}
```

**Option 2: Using absolute numbers**
```json
{
  "graph_parameters": {
    "num_weak": 3,
    "num_mandatory": 2,
    "num_discretionary": 3,
    "seed": 42
  }
}
```

### Node Capacities

Define capacity for each node with **multiple flexible options**:

#### Option 1: Default capacity for all nodes
```json
{
  "capacities": {
    "default": 10
  }
}
```
All nodes will have capacity 10.

#### Option 2: Default + Custom overrides (Recommended)
```json
{
  "capacities": {
    "default": 10,
    "custom": {
      "2": 30,
      "3": 2,
      "4": 1
    }
  }
}
```
Most nodes have capacity 10, but nodes 2, 3, and 4 have custom values. **Best for large graphs!**

#### Option 3: Random capacities
```json
{
  "capacities": {
    "random": {
      "min": 5,
      "max": 20,
      "seed": 42
    }
  }
}
```
Each node gets a random capacity between 5 and 20. Use `seed` for reproducibility.

#### Option 4: Random + Custom overrides
```json
{
  "capacities": {
    "random": {
      "min": 5,
      "max": 20,
      "seed": 42
    },
    "custom": {
      "2": 100,
      "3": 1
    }
  }
}
```
Random capacities for most nodes, but nodes 2 and 3 have fixed values.

#### Option 5: Explicit (Old format - for small graphs only)
```json
{
  "capacities": {
    "1": 10, "2": 30, "3": 2, "4": 1,
    "5": 10, "6": 4, "7": 5, "8": 5
  }
}
```
Specify each node individually. **Not recommended for graphs with many nodes.**

### Debug Options

Control visualization and logging:

```json
{
  "debug": {
    "plot_initial_graphs": false,
    "plot_intermediate": false,
    "plot_final": true,
    "save_plots": true,
    "verbose": false,
    "verbose_level2": false,
    "verbose_level3": false
  }
}
```

- `plot_initial_graphs`: Show initial graph visualizations
- `plot_intermediate`: Show intermediate step visualizations
- `plot_final`: Show final result visualization
- `save_plots`: Save plots to disk
- `verbose`: Enable detailed logging (level 1)
- `verbose_level2`: Enable more detailed logging (level 2)
- `verbose_level3`: Enable most detailed logging (level 3)

### Algorithm Configuration

```json
{
  "algorithm": {
    "weight_range": [1, 10]
  }
}
```

### Output Configuration

```json
{
  "output": {
    "plots_dir": "plots",
    "results_dir": "results"
  }
}
```

## Configuration Management

### Create Default Configuration

```bash
python config_loader.py create config.json
```

### Validate Configuration

```bash
python config_loader.py validate config.json
```

### Show Configuration

```bash
python config_loader.py show config.json
```

## Algorithm Comparison Workflow

To compare this algorithm with another algorithm on the same graph:

### Step 1: Generate and Save Configuration

```python
from graph_generator import generate_complete_config
from config_loader import load_config
import json

# Load base configuration
config = load_config('config.json')
graph_config = generate_complete_config(config)

# Save for comparison
comparison_config = {
    'graph_config': graph_config,
    'algorithm_config': config.get('algorithm', {}),
    'test_id': 'test_001',
    'seed': 42
}

with open('comparison_config.json', 'w') as f:
    json.dump(comparison_config, f, indent=2)
```

### Step 2: Run First Algorithm

```python
from tree_optimizer import run_algorithm
import json

# Load comparison configuration
with open('comparison_config.json', 'r') as f:
    config = json.load(f)

# Run algorithm 1
result1 = run_algorithm(
    graph_config=config['graph_config'],
    algorithm_config=config['algorithm_config'],
    output_dir='plots_algo1'
)

# Save results
with open('results/algo1_results.json', 'w') as f:
    json.dump({
        'test_id': config['test_id'],
        'algorithm': 'tree_optimizer',
        'execution_time': result1['execution_time'],
        'num_nodes': result1['num_nodes'],
        'num_edges': result1['num_edges']
    }, f, indent=2)
```

### Step 3: Run Second Algorithm

```python
from your_other_algorithm import run_other_algorithm
import json

# Load same configuration
with open('comparison_config.json', 'r') as f:
    config = json.load(f)

# Run algorithm 2 with same graph
result2 = run_other_algorithm(
    graph_config=config['graph_config'],
    algorithm_config=config['algorithm_config'],
    output_dir='plots_algo2'
)

# Save results
with open('results/algo2_results.json', 'w') as f:
    json.dump({
        'test_id': config['test_id'],
        'algorithm': 'other_algorithm',
        'execution_time': result2['execution_time'],
        # ... other metrics
    }, f, indent=2)
```

### Step 4: Compare Results

```python
import json

# Load both results
with open('results/algo1_results.json', 'r') as f:
    result1 = json.load(f)

with open('results/algo2_results.json', 'r') as f:
    result2 = json.load(f)

# Compare
print(f"Algorithm 1 time: {result1['execution_time']:.2f}s")
print(f"Algorithm 2 time: {result2['execution_time']:.2f}s")
```

## Capacity Configuration Examples

Here are practical examples for different use cases:

### Small Graph (8 nodes) - Default + Some Custom
```json
{
  "graph_parameters": {
    "num_nodes": 8,
    "weak_ratio": 0.4,
    "mandatory_ratio": 0.2,
    "seed": 42
  },
  "capacities": {
    "default": 10,
    "custom": {
      "2": 30,
      "3": 2
    }
  }
}
```

### Large Graph (50 nodes) - Random
```json
{
  "graph_parameters": {
    "num_nodes": 50,
    "weak_ratio": 0.3,
    "mandatory_ratio": 0.2,
    "seed": 42
  },
  "capacities": {
    "random": {
      "min": 5,
      "max": 25,
      "seed": 42
    }
  }
}
```

### Large Graph with Key Nodes - Random + Custom
```json
{
  "graph_parameters": {
    "num_nodes": 50,
    "weak_ratio": 0.3,
    "mandatory_ratio": 0.2,
    "seed": 42
  },
  "capacities": {
    "random": {
      "min": 5,
      "max": 15,
      "seed": 42
    },
    "custom": {
      "1": 100,
      "2": 100,
      "3": 1
    }
  }
}
```
Nodes 1 and 2 are high-capacity hubs, node 3 is constrained, others are random.

### Testing - All Equal
```json
{
  "graph_parameters": {
    "num_nodes": 10,
    "seed": 42
  },
  "capacities": {
    "default": 10
  }
}
```



The `main.py` file contains several usage examples:

```bash
# Run example 1: Config file usage
python main.py 1

# Run example 2: Programmatic configuration
python main.py 2

# Run example 3: Setup for comparison
python main.py 3

# Run example 4: Load and compare
python main.py 4

# Run all examples
python main.py all
```

## Output

The algorithm produces:

1. **Best Tree**: The optimal tree structure (NetworkX Graph object)
2. **Execution Time**: Total time taken to find the solution
3. **Visualizations**: PNG images of graphs (if `save_plots` is enabled)
   - Initial graphs
   - Intermediate steps (if enabled)
   - Final best tree

## Node Types

- **Weak Nodes** (green): Regular nodes with limited capabilities
- **Mandatory Power Nodes** (red): Must be included, have higher capacity
- **Discretionary Power Nodes** (orange): Optional high-capacity nodes

## Algorithm Behavior

The algorithm:
1. Creates initial graph with weak and mandatory nodes
2. Generates all valid tree combinations
3. Filters based on connectivity and hub constraints
4. Evaluates discretionary node combinations
5. Compares trees using edge costs and node degree costs
6. Returns the optimal tree

## Reproducibility

For deterministic results, always set a `seed` in the configuration:

```json
{
  "graph_parameters": {
    "seed": 42
  }
}
```

This ensures:
- Same random edge weights
- Same graph generation
- Reproducible comparisons between algorithms

## Troubleshooting

### "The graph is not fully connected"
The initial configuration produces a disconnected graph. Check node counts and connectivity requirements.

### "Missing required section"
Configuration file is incomplete. Use `python config_loader.py validate config.json` to check.

### Import errors
Ensure all required packages are installed: `pip install networkx matplotlib`

## Performance Notes

- Algorithm complexity depends on number of nodes and discretionary nodes
- Use `yield` generators to avoid memory saturation with large graphs
- For large graphs, consider disabling visualization (`plot_intermediate: false`)

## License

[Your license here]

## Contributing

[Contributing guidelines here]

## Contact

[Your contact information here]
