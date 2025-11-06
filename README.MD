tree_optimizer.py is the exaustive algorithm
# Tree Optimization Algorithm

A refactored and modular implementation of a tree optimization algorithm for graphs with different node types (weak, mandatory power, and discretionary power nodes).

## Features

- **Standalone Mode**: Run directly from command line with configuration files
- **Library Mode**: Import and use in your own Python scripts
- **Deterministic**: Support for random seeds for reproducible results
- **Configurable**: JSON-based configuration for easy parameter tuning
- **Algorithm Comparison Ready**: Designed to facilitate comparison with other algorithms

## File Structure

```
project/
├── tree_optimizer.py      # Main algorithm implementation
├── graph_generator.py     # Graph configuration generation
├── config_loader.py       # Configuration file management
├── config.json           # Configuration file
└── README.md            # This file
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

2. Run the algorithm:
```bash
python tree_optimizer.py --config config.json
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
