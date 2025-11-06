# greedy_steiner.py
"""
Greedy Steiner Tree Algorithm - Core Functions
Extracted for use as a library module
"""

import networkx as nx
import random
from itertools import combinations

# Global variables (required by the algorithm)
power_capacities = {}
main_graph = None


class Solution:
    def __init__(self, steiner_tree, capacity_usage, connected_weak, failed_connections,
                 total_cost, capacity_cost, discretionary_used, graph_info="",
                 acc_cost=0, aoc_cost=0, alpha=0.5):
        self.steiner_tree = steiner_tree
        self.capacity_usage = capacity_usage
        self.connected_weak = connected_weak
        self.failed_connections = failed_connections
        self.total_cost = total_cost
        self.capacity_cost = capacity_cost
        self.discretionary_used = discretionary_used
        self.graph_info = graph_info
        self.acc_cost = acc_cost
        self.aoc_cost = aoc_cost
        self.alpha = alpha
        self.score = self.calculate_score()

    def calculate_cost_function(self, graph, selected_edges, selected_nodes, alpha=0.5):
        """Calculate custom cost function C(G) = α * ACC + (1-α) * AOC"""
        n = len(graph.nodes())

        # Calculate ACC
        total_edge_weight = sum(graph[u][v]['weight'] for u, v in selected_edges)
        acc = total_edge_weight / (n * (n - 1)) if n > 1 else 0

        # Calculate AOC (normalized)
        total_weighted_saturation = 0
        total_weight = 0

        for node in selected_nodes:
            if node not in power_capacities:
                continue

            max_capacity = power_capacities.get(node, float('inf'))
            current_usage = self.capacity_usage.get(node, 0)

            if max_capacity == float('inf'):
                continue

            if max_capacity > 0:
                saturation = current_usage / max_capacity
            else:
                saturation = 1.0 if current_usage > 0 else 0.0

            degree = len([edge for edge in selected_edges if node in edge])
            weight = 1 + (degree / (n - 1)) if n > 1 else 1
            capped_saturation = min(saturation, 2.0)

            total_weighted_saturation += capped_saturation * weight
            total_weight += weight

        if total_weight > 0:
            aoc_raw = total_weighted_saturation / total_weight
            if aoc_raw <= 1:
                aoc = aoc_raw
            else:
                aoc = 1 - 0.5 * (2 - aoc_raw)
        else:
            aoc = 0

        aoc = max(0.0, min(1.0, aoc))
        cost = alpha * acc + (1 - alpha) * aoc

        return cost, acc, aoc

    def calculate_score(self):
        """Calculate score for solution comparison"""
        selected_nodes = set()
        selected_edges = list(self.steiner_tree.edges())

        for u, v in selected_edges:
            selected_nodes.add(u)
            selected_nodes.add(v)

        try:
            cost_func_value, acc, aoc = self.calculate_cost_function(
                main_graph, selected_edges, selected_nodes, self.alpha
            )
            self.acc_cost = acc
            self.aoc_cost = aoc
            self.weighted_cost = cost_func_value
        except:
            cost_func_value = self.total_cost / 1000
            self.acc_cost = cost_func_value
            self.aoc_cost = 0
            self.weighted_cost = cost_func_value

        connection_penalty = len(self.failed_connections) * 1000

        connectivity_penalty = 0
        if len(selected_edges) > 0:
            temp_graph = nx.Graph()
            temp_graph.add_edges_from(selected_edges)
            if not nx.is_connected(temp_graph):
                connectivity_penalty = 500

        total_score = cost_func_value * 1000 + connection_penalty + connectivity_penalty

        return total_score


def find_all_paths_to_mandatory(graph, weak_node, mandatory_nodes, discretionary_nodes, max_hops=4):
    """Find ALL possible paths from weak node to mandatory nodes"""
    all_paths = []

    # Direct paths
    for mandatory_node in mandatory_nodes:
        if graph.has_edge(weak_node, mandatory_node):
            cost = graph[weak_node][mandatory_node]['weight']
            all_paths.append({
                'path': [weak_node, mandatory_node],
                'cost': cost,
                'target_mandatory': mandatory_node,
                'discretionary_used': []
            })

    # Paths through 1 discretionary
    for disc_node in discretionary_nodes:
        if graph.has_edge(weak_node, disc_node):
            cost_to_disc = graph[weak_node][disc_node]['weight']

            for mandatory_node in mandatory_nodes:
                if graph.has_edge(disc_node, mandatory_node):
                    total_cost = cost_to_disc + graph[disc_node][mandatory_node]['weight']
                    all_paths.append({
                        'path': [weak_node, disc_node, mandatory_node],
                        'cost': total_cost,
                        'target_mandatory': mandatory_node,
                        'discretionary_used': [disc_node]
                    })

    # Paths through 2 discretionary
    if max_hops >= 3:
        for disc1 in discretionary_nodes:
            if graph.has_edge(weak_node, disc1):
                cost_to_disc1 = graph[weak_node][disc1]['weight']

                for disc2 in discretionary_nodes:
                    if disc1 != disc2 and graph.has_edge(disc1, disc2):
                        cost_disc1_to_disc2 = graph[disc1][disc2]['weight']

                        for mandatory_node in mandatory_nodes:
                            if graph.has_edge(disc2, mandatory_node):
                                total_cost = cost_to_disc1 + cost_disc1_to_disc2 + graph[disc2][mandatory_node]['weight']
                                all_paths.append({
                                    'path': [weak_node, disc1, disc2, mandatory_node],
                                    'cost': total_cost,
                                    'target_mandatory': mandatory_node,
                                    'discretionary_used': [disc1, disc2]
                                })

    all_paths.sort(key=lambda x: x['cost'])
    return all_paths


def solve_with_discretionary_subset(graph, weak_nodes, mandatory_nodes, discretionary_subset,
                                   power_capacities_copy, graph_info="", alpha=0.5):
    """Solve using specific subset of discretionary nodes"""
    steiner_tree = nx.Graph()
    capacity_usage = {node: 0 for node in mandatory_nodes + discretionary_subset}
    connected_weak = set()
    failed_connections = []
    actually_used_discretionary = set()

    # Connect all mandatory nodes first
    if len(mandatory_nodes) > 1:
        mandatory_subgraph = graph.subgraph(mandatory_nodes).copy()

        if nx.is_connected(mandatory_subgraph):
            mandatory_mst = nx.minimum_spanning_tree(mandatory_subgraph, weight='weight')
            for u, v in mandatory_mst.edges():
                steiner_tree.add_edge(u, v, weight=graph[u][v]['weight'])
        else:
            # Connect using shortest paths
            mandatory_set = set(mandatory_nodes)
            connected_mandatory = set([mandatory_nodes[0]])

            while connected_mandatory != mandatory_set:
                best_path = None
                best_cost = float('inf')

                for connected in connected_mandatory:
                    for unconnected in mandatory_set - connected_mandatory:
                        try:
                            path = nx.shortest_path(graph, connected, unconnected, weight='weight')
                            cost = nx.shortest_path_length(graph, connected, unconnected, weight='weight')

                            if cost < best_cost:
                                best_cost = cost
                                best_path = path
                        except nx.NetworkXNoPath:
                            continue

                if best_path:
                    for i in range(len(best_path) - 1):
                        steiner_tree.add_edge(best_path[i], best_path[i+1],
                                            weight=graph[best_path[i]][best_path[i+1]]['weight'])

                        if best_path[i] in discretionary_subset:
                            capacity_usage[best_path[i]] = capacity_usage.get(best_path[i], 0) + 1
                            actually_used_discretionary.add(best_path[i])
                        if best_path[i+1] in discretionary_subset:
                            capacity_usage[best_path[i+1]] = capacity_usage.get(best_path[i+1], 0) + 1
                            actually_used_discretionary.add(best_path[i+1])

                    for node in best_path:
                        if node in mandatory_set:
                            connected_mandatory.add(node)
                else:
                    break

    # Find paths for weak nodes
    all_weak_options = {}
    for weak_node in weak_nodes:
        paths = find_all_paths_to_mandatory(graph, weak_node, mandatory_nodes, discretionary_subset)
        all_weak_options[weak_node] = paths

    # Greedy connection with cost function
    all_options = []
    for weak_node, paths in all_weak_options.items():
        for path_info in paths:
            path_edges = [(path_info['path'][i], path_info['path'][i+1])
                         for i in range(len(path_info['path'])-1)]

            simulated_capacity_usage = capacity_usage.copy()
            simulated_tree_edges = list(steiner_tree.edges())

            target_mandatory = path_info['target_mandatory']
            discretionary_used = path_info['discretionary_used']
            path = path_info['path']

            simulated_capacity_usage[target_mandatory] = simulated_capacity_usage.get(target_mandatory, 0) + 1
            for disc_node in discretionary_used:
                simulated_capacity_usage[disc_node] = simulated_capacity_usage.get(disc_node, 0) + 1

            for i in range(len(path) - 1):
                simulated_tree_edges.append((path[i], path[i+1]))

            simulated_selected_nodes = set()
            for u, v in simulated_tree_edges:
                simulated_selected_nodes.add(u)
                simulated_selected_nodes.add(v)

            edge_weight_sum = sum(graph[u][v]['weight'] for u, v in path_edges)
            n = len(graph.nodes())
            incremental_acc = edge_weight_sum / (n * (n - 1)) if n > 1 else 0

            # Calculate AOC increment (simplified)
            incremental_aoc = 0

            incremental_cost = alpha * incremental_acc + (1 - alpha) * incremental_aoc

            all_options.append({
                'weak_node': weak_node,
                'incremental_cost': incremental_cost,
                'incremental_acc': incremental_acc,
                'incremental_aoc': incremental_aoc,
                'edge_cost': edge_weight_sum,
                **path_info
            })

    all_options.sort(key=lambda x: x['incremental_cost'])

    # Connect weak nodes
    for option in all_options:
        weak_node = option['weak_node']

        if weak_node in connected_weak:
            continue

        path = option['path']
        target_mandatory = option['target_mandatory']
        discretionary_used = option['discretionary_used']

        capacity_usage[target_mandatory] += 1
        for disc_node in discretionary_used:
            capacity_usage[disc_node] += 1
            actually_used_discretionary.add(disc_node)

        for i in range(len(path) - 1):
            steiner_tree.add_edge(path[i], path[i+1], weight=graph[path[i]][path[i+1]]['weight'])

        connected_weak.add(weak_node)

    # Handle remaining weak nodes
    remaining_weak = set(weak_nodes) - connected_weak

    if remaining_weak:
        for weak_node in remaining_weak:
            available_paths = all_weak_options.get(weak_node, [])
            if available_paths:
                chosen_path = available_paths[0]
                target_mandatory = chosen_path['target_mandatory']
                discretionary_used = chosen_path['discretionary_used']
                path = chosen_path['path']

                capacity_usage[target_mandatory] += 1
                for disc_node in discretionary_used:
                    capacity_usage[disc_node] += 1
                    actually_used_discretionary.add(disc_node)

                for i in range(len(path) - 1):
                    steiner_tree.add_edge(path[i], path[i+1], weight=graph[path[i]][path[i+1]]['weight'])

                connected_weak.add(weak_node)
            else:
                failed_connections.append(weak_node)

    # Calculate statistics
    total_cost = sum(graph[u][v]['weight'] for u, v in steiner_tree.edges())

    capacity_cost = 0
    nodes_actually_used = [n for n in capacity_usage if capacity_usage[n] > 0 and power_capacities_copy.get(n, 0) > 0]

    if nodes_actually_used:
        capacity_ratios = []
        for node in nodes_actually_used:
            if power_capacities_copy[node] > 0:
                ratio = capacity_usage[node] / power_capacities_copy[node]
                capacity_ratios.append(ratio)

        capacity_cost = sum(capacity_ratios) / len(capacity_ratios)

    actually_used_list = sorted(list(actually_used_discretionary))

    return Solution(steiner_tree, capacity_usage, connected_weak, failed_connections,
                   total_cost, capacity_cost, actually_used_list, graph_info, alpha=alpha)


def find_best_solution_simplified(graph, weak_nodes, mandatory_nodes, all_discretionary_nodes,
                                 power_capacities_copy, alpha=0.5):
    """Find best solution by testing without and with discretionary nodes"""
    global main_graph, power_capacities
    main_graph = graph
    power_capacities = power_capacities_copy

    all_solutions = []

    # Solution without discretionary
    solution_no_disc = solve_with_discretionary_subset(
        graph, weak_nodes, mandatory_nodes, [], power_capacities_copy.copy(),
        "WITHOUT discretionary", alpha
    )
    all_solutions.append(solution_no_disc)

    # Solution with all discretionary
    solution_all_disc = solve_with_discretionary_subset(
        graph, weak_nodes, mandatory_nodes, all_discretionary_nodes, power_capacities_copy.copy(),
        f"WITH ALL discretionary", alpha
    )
    all_solutions.append(solution_all_disc)

    # Sort by score
    all_solutions.sort(key=lambda s: s.score)
    best_solution = all_solutions[0]

    return best_solution, all_solutions
