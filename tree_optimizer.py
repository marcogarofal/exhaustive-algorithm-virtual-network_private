# tree_optimizer.py
"""
Tree Optimization Algorithm - Main Implementation
Maintains exact same functionality as original code while being callable as library
"""

import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, chain
import random
from collections import Counter
import copy
import time
import os


class DebugConfig:
    """Debug configuration to replace global debug variables"""
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = {}
        self.plot_initial_graphs = config_dict.get('plot_initial_graphs', False)
        self.plot_intermediate = config_dict.get('plot_intermediate', False)
        self.plot_final = config_dict.get('plot_final', False)
        self.save_plots = config_dict.get('save_plots', False)
        self.verbose = config_dict.get('verbose', False)
        self.verbose_level2 = config_dict.get('verbose_level2', False)
        self.verbose_level3 = config_dict.get('verbose_level3', False)


class CombinationGraph:
    def __init__(self, graph, weak_nodes, debug_config):
        if not nx.is_connected(graph):
            raise ValueError("\tThe graph is not fully connected.")
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.weak_nodes = weak_nodes
        self.debug_config = debug_config
        self.all_possible_links = list(self._generate_combinations_mod(graph.nodes, 2))
        self.all_combinations = list(combinations(self.all_possible_links, self.num_nodes - 1))
        print("\t...generating")

    def _generate_combinations_mod(self, elements, r):
        """Generator for combinations, filtering weak-weak connections"""
        for combo in combinations(elements, r):
            if self.debug_config.verbose:
                print(combo)
            if combo[0] in self.weak_nodes and combo[1] in self.weak_nodes:
                pass
            else:
                yield combo

    def remove_trees_where_weak_nodes_are_hubs(self, weak_nodes, list_tree):
        valori = list(chain(*list_tree))
        count = Counter(valori)
        for weak_node in weak_nodes:
            if count[weak_node] > 1:
                return False
        return True

    def are_discretionary_nodes_singularly_connected(self, edges, discretionary_nodes, check_only_discretionary, check_no_mandatory):
        if check_only_discretionary:
            return True
        # ✅ Removed check_no_mandatory bypass - discretionary nodes must ALWAYS have degree >= 2
        # They should act as bridges/hubs, not as leaf nodes
        graph = {}
        for edge in edges:
            u, v = edge
            if u not in graph:
                graph[u] = []
            if v not in graph:
                graph[v] = []
            graph[u].append(v)
            graph[v].append(u)
        return all(len(graph[node]) > 1 for node in discretionary_nodes if node in graph)

    def filter_combinations_discretionary(self, weak_nodes, power_nodes_mandatory, power_nodes_discretionary, check_only_discretionary, check_no_mandatory):
        list_nodes_graph = self.graph.nodes()
        valid_combinations = []
        print("\t...filtering")

        for combination in self.all_combinations:
            if combination:
                if power_nodes_discretionary is not None:
                    if self.remove_trees_where_weak_nodes_are_hubs(weak_nodes, combination):
                        if self.debug_config.verbose:
                            print("\tcheck C")
                        if self.are_discretionary_nodes_singularly_connected(combination, power_nodes_discretionary, check_only_discretionary, check_no_mandatory):
                            if self.debug_config.verbose:
                                print("\tcheck D")
                            if is_connected(combination, len(list_nodes_graph)):
                                print("\tcheck E")
                                valid_combinations.append(combination)
                        else:
                            pass
                else:
                    if self.remove_trees_where_weak_nodes_are_hubs(weak_nodes, combination):
                        print("\t\tcheck CC")
                        if is_connected(combination, len(list_nodes_graph)):
                            print("\t\tcheck EE")
                            valid_combinations.append(combination)
                    else:
                        print("\t\tweak_hubs")
                        pass
        print("valid_combination:", valid_combinations)
        self.all_combinations = valid_combinations


def is_connected(graph, number_of_nodes):
    """Verify if the graph is connected using DFS"""
    def dfs_modified(graph, start):
        visited = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                neighbours = [neighbour for edge in graph for neighbour in edge if node in edge and neighbour != node]
                stack.extend(neighbour for neighbour in neighbours if neighbour not in visited)
        return visited

    start_node = graph[0][0]
    reachable_nodes = dfs_modified(graph, start_node)
    return len(reachable_nodes) == number_of_nodes


def create_graph(weak_nodes=None, power_nodes_mandatory=None, power_nodes_discretionary=None, seed=None):
    """Create a complete graph with random weights"""
    if seed is not None:
        random.seed(seed)

    G = nx.Graph()
    all_nodes = []

    if (weak_nodes is not None) and (power_nodes_mandatory is not None) and (power_nodes_discretionary is not None):
        G.add_nodes_from(weak_nodes, node_type='weak')
        G.add_nodes_from(power_nodes_mandatory, node_type='power_mandatory')
        G.add_nodes_from(power_nodes_discretionary, node_type='power_discretionary')
        all_nodes = list(weak_nodes) + list(power_nodes_mandatory) + list(power_nodes_discretionary)
    elif (weak_nodes is not None) and (power_nodes_mandatory is not None):
        G.add_nodes_from(weak_nodes, node_type='weak')
        G.add_nodes_from(power_nodes_mandatory, node_type='power_mandatory')
        all_nodes = list(weak_nodes) + list(power_nodes_mandatory)
    elif (weak_nodes is not None) and (power_nodes_discretionary is not None):
        G.add_nodes_from(weak_nodes, node_type='weak')
        G.add_nodes_from(power_nodes_discretionary, node_type='power_discretionary')
        all_nodes = list(weak_nodes) + list(power_nodes_discretionary)
    elif power_nodes_discretionary is not None:
        G.add_nodes_from(power_nodes_discretionary, node_type='power_discretionary')
        all_nodes = list(power_nodes_discretionary)
    else:
        print("\tnot a possible case")

    for i in all_nodes:
        for j in all_nodes:
            if i != j and not G.has_edge(i, j):
                weight = random.randint(1, 10)
                G.add_edge(i, j, weight=weight)
    return G


def draw_graph(G):
    """Draw graph with colored nodes"""
    plt.clf()
    pos = nx.spring_layout(G)
    node_colors = {'weak': 'green', 'power_mandatory': 'red', 'power_discretionary': 'orange'}
    colors = [node_colors[data['node_type']] for _, data in G.nodes(data=True)]
    edge_labels = {(i, j): G[i][j]['weight'] for i, j in G.edges()}

    nx.draw(G, pos, with_labels=True, node_color=colors, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()


def save_graph(G, path, count_picture, name=None, edge_cost=None, degree_cost=None):
    """Save graph to file with optional score annotation"""
    pos = nx.spring_layout(G)
    node_colors = {'weak': 'green', 'power_mandatory': 'red', 'power_discretionary': 'orange'}
    colors = [node_colors[data['node_type']] for _, data in G.nodes(data=True)]
    edge_labels = {(i, j): G[i][j]['weight'] for i, j in G.edges()}

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    nx.draw(G, pos, with_labels=True, node_color=colors, font_weight='bold', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    # Add score annotation only if provided (for best_tree only)
    if edge_cost is not None and degree_cost is not None:
        total_cost = edge_cost + degree_cost
        score_text = f'Edge Cost: {edge_cost:.4f}\nDegree Cost: {degree_cost:.4f}\nTotal: {total_cost:.4f}'
        ax.text(0.02, 0.98, score_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        # Include score in filename only for best_tree
        filename = f"{count_picture}_{name}_e{edge_cost:.4f}_d{degree_cost:.4f}_t{total_cost:.4f}.png"
    else:
        # No score in filename for intermediate trees
        if name is None:
            filename = f"{count_picture}_graph.png"
        else:
            filename = f"{count_picture}_{name}.png"

    path_to_save = os.path.join(path, filename)
    plt.savefig(path_to_save)
    plt.close()
    return count_picture + 1, filename


def build_tree_from_list_edges(G, desired_edges, no_plot=None):
    """Build tree from list of edges"""
    G_copy = copy.deepcopy(G)
    edges_to_remove = [edge for edge in G_copy.edges() if edge not in desired_edges]
    G_copy.remove_edges_from(edges_to_remove)

    if no_plot is False or no_plot is None:
        pos = nx.spring_layout(G_copy)
        node_colors = {'weak': 'green', 'power_mandatory': 'red', 'power_discretionary': 'orange'}
        colors = [node_colors[data['node_type']] for _, data in G_copy.nodes(data=True)]
        edge_labels = {(i, j): G_copy[i][j].get('weight', None) for i, j in G_copy.edges()}

        nx.draw(G_copy, pos, with_labels=True, node_color=colors, font_weight='bold')
        nx.draw_networkx_edge_labels(G_copy, pos, edge_labels=edge_labels)
        nx.draw_networkx_edges(G_copy, pos, edgelist=desired_edges)
        plt.show()
    return G_copy


def get_weight(item):
    """Helper function for max weight extraction"""
    return item[1]['weight']


def save_scores_to_json(scores, output_path):
    """Save scores dictionary to JSON file"""
    import json
    with open(output_path, 'w') as f:
        json.dump(scores, f, indent=2)
    print(f"Scores saved to {output_path}")


def calculate_absolute_scores(trees_data, capacities, power_nodes_mandatory, power_nodes_discretionary):
    """
    Calculate absolute scores for all trees with global normalization.

    Args:
        trees_data: list of (tree, filename) tuples
        capacities: node capacities dict
        power_nodes_mandatory: list of mandatory power nodes
        power_nodes_discretionary: list of discretionary power nodes

    Returns:
        dict with scores for each tree
    """
    if not trees_data:
        return {}

    # Find global max edge weight and max edges across ALL trees
    global_max_weight = 0
    global_max_edges = 0

    for tree, _ in trees_data:
        if tree and tree.number_of_edges() > 0:
            edges_with_weights = [(edge, tree.get_edge_data(edge[0], edge[1])) for edge in tree.edges()]
            tree_max_weight = max(edges_with_weights, key=get_weight)[1]["weight"]
            tree_num_edges = len(edges_with_weights)

            global_max_weight = max(global_max_weight, tree_max_weight)
            global_max_edges = max(global_max_edges, tree_num_edges)

    if global_max_weight == 0:
        global_max_weight = 1  # Avoid division by zero
    if global_max_edges == 0:
        global_max_edges = 1  # Avoid division by zero

    # Calculate scores for each tree using global normalization
    scores = {
        "global_max_weight": global_max_weight,
        "global_max_edges": global_max_edges,
        "trees": {}
    }

    set_power_nodes = set(list(power_nodes_mandatory) + list(power_nodes_discretionary))

    for tree, filename in trees_data:
        if tree and tree.number_of_edges() > 0:
            # Calculate edge cost with GLOBAL normalization
            edges_with_weights = [(edge, tree.get_edge_data(edge[0], edge[1])) for edge in tree.edges()]
            total_weight = sum(data['weight'] for edge, data in edges_with_weights)

            # Use global max_weight and global max_edges for normalization
            edge_cost = total_weight / (global_max_weight * global_max_edges)

            # Calculate degree cost
            degree_cost = 0
            for node in tree.nodes():
                if node in set_power_nodes and node in capacities:
                    degree_cost += tree.degree(node) / capacities[node]
            degree_cost = degree_cost / len(tree.nodes())

            total_cost = edge_cost + degree_cost

            scores["trees"][filename] = {
                "edge_cost": round(edge_cost, 4),
                "degree_cost": round(degree_cost, 4),
                "total": round(total_cost, 4),
                "num_nodes": tree.number_of_nodes(),
                "num_edges": tree.number_of_edges(),
                "nodes": list(tree.nodes()),
                "edges": [list(edge) for edge in tree.edges()]
            }
        else:
            scores["trees"][filename] = {
                "edge_cost": 0,
                "degree_cost": 0,
                "total": 0,
                "num_nodes": 0,
                "num_edges": 0,
                "nodes": [],
                "edges": []
            }

    return scores


def compare_2_trees(tree1, tree2, power_nodes_mandatory, power_nodes_discretionary, capacities, debug_config):
    """Compare two trees and return the best one with its costs"""
    if tree1 is None:
        tree1 = tree2

    print("\t\tcompare_2_trees")

    edges_with_weights1 = [(edge, tree1.get_edge_data(edge[0], edge[1])) for edge in tree1.edges()]
    max_edge_cost1 = max(edges_with_weights1, key=get_weight)[1]["weight"]
    edgecost1 = 0

    edges_with_weights2 = [(edge, tree2.get_edge_data(edge[0], edge[1])) for edge in tree2.edges()]
    max_edge_cost2 = max(edges_with_weights2, key=get_weight)[1]["weight"]
    edgecost2 = 0

    set_power_nodes = set(list(power_nodes_mandatory) + list(power_nodes_discretionary))

    cost_degree1 = 0
    for x in tree1.nodes():
        if x in set_power_nodes:
            try:
                cost_degree1 += tree1.degree(x) / capacities[x]
            except (AttributeError, KeyError):
                print("error in cost_degree1")

    cost_degree2 = 0
    for x in tree2.nodes():
        if x in set_power_nodes:
            try:
                cost_degree2 += tree2.degree(x) / capacities[x]
            except (AttributeError, KeyError):
                print("error in cost_degree2")

    if debug_config.verbose:
        print("cost_degree_:", cost_degree1, cost_degree2)
        print("cost_degree:", cost_degree1/len(tree1.nodes()), cost_degree2/len(tree2.nodes()))

    cost_degree1 = cost_degree1 / len(tree1.nodes())
    cost_degree2 = cost_degree2 / len(tree2.nodes())

    if max_edge_cost1 >= max_edge_cost2:
        if debug_config.verbose:
            print("\tcase max1>max2, len:", len(edges_with_weights1), " :", edges_with_weights1)
        for edge1, data1 in edges_with_weights1:
            edgecost1 += data1['weight']
        if debug_config.verbose:
            print("\t\t-edgecost1:", edgecost1, " max1:", max_edge_cost1, " numbernodes1:", len(tree1.nodes()))
        edgecost1 = edgecost1 / (max_edge_cost1 * len(edges_with_weights1))

        for edge2, data2 in edges_with_weights2:
            edgecost2 += data2['weight']
        if debug_config.verbose:
            print("\t\t-edgecost2:", edgecost2, " max1:", max_edge_cost1, " numbernodes2:", len(tree2.nodes()))
        edgecost2 = edgecost2 / (max_edge_cost1 * len(edges_with_weights1))
    else:
        if debug_config.verbose:
            print("\tcase max2>max1, len:", len(edges_with_weights2), " :", edges_with_weights2)
        for edge1, data1 in edges_with_weights1:
            edgecost1 += data1['weight']
        if debug_config.verbose:
            print("\t\t-edgecost1:", edgecost1, " max1", max_edge_cost1, " numbernodes1:", len(tree1.nodes()))
        edgecost1 = edgecost1 / (max_edge_cost2 * len(edges_with_weights2))

        for edge2, data2 in edges_with_weights2:
            edgecost2 += data2['weight']
        if debug_config.verbose:
            print("\t\t-edgecost2:", edgecost2, " max2", max_edge_cost2, " numbernodes2:", len(tree2.nodes()))
        edgecost2 = edgecost2 / (max_edge_cost2 * len(edges_with_weights2))

    if debug_config.verbose:
        print("\nedge:", edgecost1, edgecost2, "\ndegree:", cost_degree1, cost_degree2, "\nsum:", edgecost1+cost_degree1, edgecost2+cost_degree2)

    if edgecost1 + cost_degree1 <= edgecost2 + cost_degree2:
        if debug_config.verbose:
            print("\n\n\nbest_tree")
        # Return tree1 with its costs (tree1 is better or equal)
        return tree1, edgecost1, cost_degree1, True
    else:
        if debug_config.verbose:
            print("\n\n\nnew_tree")
        # Return tree2 with its costs (tree2 is better)
        return tree2, edgecost2, cost_degree2, False


def join_2_trees(graph1, graph2, weak_nodes, power_nodes_mandatory, power_nodes_discretionary, added_edges, seed=None, pre_built_graph=None):
    """Join two trees into one graph"""
    if seed is not None:
        random.seed(seed)

    G = nx.Graph()
    all_nodes = []

    if (weak_nodes is not None) and (power_nodes_mandatory is not None) and (power_nodes_discretionary is not None):
        G.add_nodes_from(weak_nodes, node_type='weak')
        G.add_nodes_from(power_nodes_mandatory, node_type='power_mandatory')
        G.add_nodes_from(power_nodes_discretionary, node_type='power_discretionary')
        all_nodes = list(weak_nodes) + list(power_nodes_mandatory) + list(power_nodes_discretionary)
    elif (weak_nodes is not None) and (power_nodes_mandatory is not None):
        G.add_nodes_from(weak_nodes, node_type='weak')
        G.add_nodes_from(power_nodes_mandatory, node_type='power_mandatory')
        all_nodes = list(weak_nodes) + list(power_nodes_mandatory)
    elif power_nodes_discretionary is not None:
        G.add_nodes_from(power_nodes_discretionary, node_type='power_discretionary')
        all_nodes = list(power_nodes_discretionary)
    else:
        print("\tnot a possible case2")

    for i in all_nodes:
        for j in all_nodes:
            if i != j and not G.has_edge(i, j):
                matching_tuple = next((tup for tup in added_edges if set(tup[:2]) == set((i, j))), None)

                # Check pre-built graph first
                if pre_built_graph is not None and pre_built_graph.has_edge(i, j):
                    G.add_edge(i, j, weight=pre_built_graph[i][j]['weight'])
                elif (graph1.has_edge(i, j) or graph1.has_edge(j, i)):
                    G.add_edge(i, j, weight=graph1[i][j]['weight'])
                elif (graph2.has_edge(i, j) or graph2.has_edge(j, i)):
                    G.add_edge(i, j, weight=graph2[i][j]['weight'])
                elif matching_tuple:
                    weight_value = next(iter(matching_tuple[2]))
                    G.add_edge(i, j, weight=weight_value)
                else:
                    weight = random.randint(1, 10)
                    G.add_edge(i, j, weight=weight)
                    new_element = (i, j, frozenset({weight}))
                    added_edges.add(new_element)
    return G


def generate_graphs(graph, power_nodes_discretionary, weak_nodes, power_nodes_mandatory, added_edges, debug_config, seed=None, pre_built_graph=None):
    """Generator that yields graphs with different discretionary node combinations"""
    print("\n2nd graph")

    if pre_built_graph is not None:
        # Use pre-built graph for discretionary phase
        graph2 = pre_built_graph.subgraph(power_nodes_discretionary).copy()
    else:
        graph2 = create_graph(power_nodes_discretionary=power_nodes_discretionary, seed=seed)

    if debug_config.plot_initial_graphs:
        draw_graph(graph2)

    def generate_combinations(elements):
        number_of_nodes = len(elements)
        for x in range(1, number_of_nodes + 1):
            for combo in combinations(elements, x):
                yield combo

    combinations_only_power_nodes_discretionary = generate_combinations(graph2.nodes)
    count = 0

    for combo in combinations_only_power_nodes_discretionary:
        if combo:
            lista_risultante = []
            for coppia in combo:
                try:
                    lista_risultante.extend(coppia)
                except:
                    lista_risultante.append(coppia)

            if count == 0:
                graph3 = join_2_trees(graph, graph2, weak_nodes=weak_nodes,
                                     power_nodes_mandatory=power_nodes_mandatory,
                                     power_nodes_discretionary=lista_risultante,
                                     added_edges=added_edges, seed=seed, pre_built_graph=pre_built_graph)
                graph3_bak = graph3
            else:
                graph3 = join_2_trees(graph3_bak, graph2, weak_nodes=weak_nodes,
                                     power_nodes_mandatory=power_nodes_mandatory,
                                     power_nodes_discretionary=lista_risultante,
                                     added_edges=added_edges, seed=seed, pre_built_graph=pre_built_graph)

            count += 1
            print("\n\tgraph3_provv")
            yield graph3


def process_graph(graph, weak_nodes, power_nodes_mandatory, power_nodes_discretionary,
                 best_tree, best_edge_cost, best_degree_cost, capacities,
                 check_only_discretionary, check_no_mandatory,
                 debug_config, plot_path, count_picture, trees_registry):
    """Process a graph to find optimal tree"""
    if debug_config.plot_intermediate:
        draw_graph(graph)

    combinations_graph = CombinationGraph(graph, weak_nodes, debug_config)
    if debug_config.verbose:
        input("\n\nENTER to continue...")

    combinations_graph.filter_combinations_discretionary(weak_nodes, power_nodes_mandatory,
                                                        power_nodes_discretionary,
                                                        check_only_discretionary, check_no_mandatory)
    print("here4:", len(combinations_graph.all_combinations))
    if debug_config.verbose:
        input("\n\nENTER to continue...")

    for x in combinations_graph.all_combinations:
        if debug_config.verbose:
            print("nodes: ", x)

        if debug_config.plot_intermediate:
            tree = build_tree_from_list_edges(graph, x, no_plot=False)
        else:
            tree = build_tree_from_list_edges(graph, x, no_plot=True)

        # Compare and get scores with is_best flag
        best_tree, best_edge_cost, best_degree_cost, is_best = compare_2_trees(
            best_tree, tree, power_nodes_mandatory,
            power_nodes_discretionary, capacities, debug_config)

        # Save intermediate tree WITHOUT score
        if debug_config.save_plots:
            count_picture, filename = save_graph(tree, plot_path, count_picture, "intermediate")
            # Register tree for later score calculation
            trees_registry.append((copy.deepcopy(tree), filename))

        if debug_config.verbose:
            input("\n\nENTER to continue...")
        if debug_config.plot_intermediate:
            draw_graph(best_tree)

    return best_tree, best_edge_cost, best_degree_cost, count_picture


def run_algorithm(graph_config, algorithm_config, debug_config=None, output_dir='plots', pre_built_graph=None):
    """
    Main entry point for the tree optimization algorithm.

    Args:
        graph_config: dict with keys: weak_nodes, power_nodes_mandatory,
                     power_nodes_discretionary, capacities
        algorithm_config: dict with keys: seed, weight_range (optional)
        debug_config: DebugConfig object or dict
        output_dir: directory for plot outputs
        pre_built_graph: optional pre-built NetworkX graph (for benchmarks)
                        If provided, skips graph creation

    Returns:
        dict with keys: best_tree, edge_cost, degree_cost, execution_time
    """
    start_time = time.time()

    # Setup debug config
    if debug_config is None:
        debug_config = DebugConfig()
    elif isinstance(debug_config, dict):
        debug_config = DebugConfig(debug_config)

    # Extract configuration
    weak_nodes = graph_config['weak_nodes']
    power_nodes_mandatory = graph_config['power_nodes_mandatory']
    power_nodes_discretionary = graph_config['power_nodes_discretionary']
    capacities = graph_config['capacities']
    seed = algorithm_config.get('seed', None)

    # Setup output directory
    plot_path = output_dir
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    else:
        for filename in os.listdir(plot_path):
            file_path = os.path.join(plot_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Unable to delete {file_path}: {e}")

    count_picture = 0
    added_edges = set()

    # Registry to keep track of all evaluated trees for final score calculation
    trees_registry = []

    # Check special cases
    num_nodes = len(weak_nodes) + len(power_nodes_mandatory) + len(power_nodes_discretionary)
    check_only_discretionary = len(power_nodes_discretionary) == num_nodes
    check_no_mandatory = len(power_nodes_mandatory) == 0 or len(power_nodes_mandatory) == 1

    # Build initial graph (or use pre-built)
    if len(weak_nodes) + len(power_nodes_mandatory) > 0:
        print("1st graph")

        if pre_built_graph is not None:
            # Use pre-built graph (for benchmarks)
            # Extract only weak and mandatory nodes for first phase
            graph = pre_built_graph.subgraph(list(weak_nodes) + list(power_nodes_mandatory)).copy()
        else:
            # Create graph normally
            graph = create_graph(weak_nodes, power_nodes_mandatory, power_nodes_discretionary=None, seed=seed)

        if debug_config.plot_initial_graphs:
            draw_graph(graph)
        if debug_config.save_plots:
            count_picture, filename = save_graph(graph, plot_path, count_picture)
            # Don't register initial graph, only trees

        combinations_graph = CombinationGraph(graph, weak_nodes, debug_config)
        combinations_graph.filter_combinations_discretionary(weak_nodes, power_nodes_mandatory, None,
                                                            check_only_discretionary, check_no_mandatory)
        print("\there3:", len(combinations_graph.all_combinations))

        if len(combinations_graph.all_combinations) > 0:
            best_tree = build_tree_from_list_edges(graph, combinations_graph.all_combinations[0], no_plot=True)
            # Initialize with first tree's costs
            edges_with_weights = [(edge, best_tree.get_edge_data(edge[0], edge[1])) for edge in best_tree.edges()]
            if edges_with_weights:
                max_edge_cost = max(edges_with_weights, key=get_weight)[1]["weight"]
                best_edge_cost = sum(data['weight'] for edge, data in edges_with_weights) / (max_edge_cost * len(edges_with_weights))
            else:
                best_edge_cost = 0
            set_power_nodes = set(list(power_nodes_mandatory) + list(power_nodes_discretionary))
            best_degree_cost = sum(best_tree.degree(x) / capacities[x] for x in best_tree.nodes() if x in set_power_nodes)
            best_degree_cost = best_degree_cost / len(best_tree.nodes()) if len(best_tree.nodes()) > 0 else 0
        else:
            print("empty list")
            best_tree = None
            best_edge_cost = 0
            best_degree_cost = 0

        for x in combinations_graph.all_combinations:
            if debug_config.verbose:
                print("nodes: ", x)

            if debug_config.plot_intermediate:
                tree = build_tree_from_list_edges(graph, x, no_plot=False)
            else:
                tree = build_tree_from_list_edges(graph, x, no_plot=True)

            # Compare and get scores with is_best flag
            best_tree, best_edge_cost, best_degree_cost, is_best = compare_2_trees(
                best_tree, tree, power_nodes_mandatory,
                power_nodes_discretionary, capacities, debug_config)

            # Save intermediate tree WITHOUT score
            if debug_config.save_plots:
                count_picture, filename = save_graph(tree, plot_path, count_picture, "first_phase")
                # Register tree for later score calculation
                trees_registry.append((copy.deepcopy(tree), filename))

            if debug_config.verbose:
                input("Enter...")
            if debug_config.plot_intermediate:
                draw_graph(best_tree)
    else:
        graph = nx.Graph()
        best_tree = None
        best_edge_cost = 0
        best_degree_cost = 0

    # Process discretionary nodes
    graphs = generate_graphs(graph, power_nodes_discretionary, weak_nodes, power_nodes_mandatory,
                            added_edges, debug_config, seed=seed, pre_built_graph=pre_built_graph)

    for graph_iter in graphs:
        best_tree, best_edge_cost, best_degree_cost, count_picture = process_graph(
            graph_iter, weak_nodes, power_nodes_mandatory,
            power_nodes_discretionary, best_tree, best_edge_cost, best_degree_cost,
            capacities, check_only_discretionary, check_no_mandatory,
            debug_config, plot_path, count_picture, trees_registry)

    end_time = time.time()
    elapsed_time = end_time - start_time

    if debug_config.plot_final:
        print("\n\n\n", best_tree)
        draw_graph(best_tree)

    # Save best tree WITH score annotation
    best_filename = None
    if debug_config.save_plots and best_tree is not None:
        count_picture, best_filename = save_graph(best_tree, plot_path, count_picture, name="best_tree",
                  edge_cost=best_edge_cost, degree_cost=best_degree_cost)
        # Register best tree
        trees_registry.append((copy.deepcopy(best_tree), best_filename))

    # Calculate absolute scores for all trees with global normalization
    if debug_config.save_plots and trees_registry:
        scores = calculate_absolute_scores(trees_registry, capacities,
                                          power_nodes_mandatory, power_nodes_discretionary)
        # Mark best tree in scores
        if best_tree is not None and best_filename and best_filename in scores["trees"]:
            scores["trees"][best_filename]["is_best"] = True

        scores_path = os.path.join(plot_path, "scores.json")
        save_scores_to_json(scores, scores_path)

    print(f"Running time: {elapsed_time} seconds")

    return {
        'best_tree': best_tree,
        'execution_time': elapsed_time,
        'num_nodes': best_tree.number_of_nodes() if best_tree else 0,
        'num_edges': best_tree.number_of_edges() if best_tree else 0
    }


if __name__ == "__main__":
    import sys

    # Standalone mode - use default configuration or load from file
    if len(sys.argv) > 1 and sys.argv[1] == '--config':
        from config_loader import load_config
        from graph_generator import generate_complete_config

        config = load_config(sys.argv[2] if len(sys.argv) > 2 else 'config.json')

        # Generate complete graph configuration from config file
        graph_config = generate_complete_config(config)

        algorithm_config = config.get('algorithm', {})
        debug_dict = config.get('debug', {})
        output_dir = config.get('output', {}).get('plots_dir', 'plots')

        result = run_algorithm(graph_config, algorithm_config, debug_dict, output_dir)
    else:
        # Default standalone configuration
        num_nodes = 8
        num_weak_nodes = int(0.4 * num_nodes)
        num_power_nodes_mandatory = int(0.2 * num_nodes)
        num_power_nodes_discretionary = num_nodes - num_weak_nodes - num_power_nodes_mandatory

        weak_nodes = list(range(1, num_weak_nodes + 1))
        power_nodes_mandatory = list(range(num_weak_nodes + 1, num_weak_nodes + num_power_nodes_mandatory + 1))
        power_nodes_discretionary = list(range(num_weak_nodes + num_power_nodes_mandatory + 1,
                                              num_weak_nodes + num_power_nodes_mandatory + num_power_nodes_discretionary + 1))

        capacities = {i: 10 for i in range(1, 21)}
        capacities.update({2: 30, 3: 2, 4: 1, 6: 4, 7: 5, 8: 5})

        graph_config = {
            'weak_nodes': weak_nodes,
            'power_nodes_mandatory': power_nodes_mandatory,
            'power_nodes_discretionary': power_nodes_discretionary,
            'capacities': capacities
        }
        algorithm_config = {'seed': 42}
        debug_dict = {'plot_final': True}

        result = run_algorithm(graph_config, algorithm_config, debug_dict)
        print(f"\nResult: {result}")
