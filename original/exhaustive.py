import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import random
from collections import Counter
from itertools import chain
import copy
import time
import os
import sys

debug0=True #plot graph1 and graph2
debug=False
debug2=False
debug3=False
debug_plot_graph=False
debug_save=False


#path plots
script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, 'plots/')
if not os.path.exists(path):
    os.makedirs(path)
else:
    if os.path.exists(path) and os.path.isdir(path):
        # Cancella il contenuto della cartella
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Delete files within the folder.
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Delete sub-folders within the folder.
            except Exception as e:
                print(f"Unable to delete {file_path}: {e}")
        print("Folder contents successfully cleared.")
    else:
        print("The folder does not exist or is not a folder.")



class CombinationGraph:
    def __init__(self, graph):
        if not nx.is_connected(graph):
            raise ValueError("\tThe graph is not fully connected.")
        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.all_possible_links = generate_combinations2_mod(graph.nodes, 2)
        self.all_combinations = generate_combinations2(self.all_possible_links, self.num_nodes - 1)
        print("\t...generating")

    def remove_trees_where_weak_nodes_are_hubs(self, weak_nodes, list_tree):
        valori = list(chain(*list_tree))
        count=Counter(valori)
        for weak_node in weak_nodes:
            if count[weak_node]>1:
                return False
        return True
    
    def are_discretionary_nodes_singularly_connected(self, edges, discretionary_nodes):
        if check_only_discretionary:
            return True
        elif check_no_mandatory:
            return True
        else:
            graph = {}
            # Build the graph represented as an adjacency dictionary
            for edge in edges:
                u, v = edge
                if u not in graph:
                    graph[u] = []
                if v not in graph:
                    graph[v] = []
                graph[u].append(v)
                graph[v].append(u)
            return all(len(graph[node]) > 1 for node in discretionary_nodes if node in graph)

    def filter_combinations_discretionary(self, weak_nodes,  power_nodes_mandatory, power_nodes_discretionary=None):
        list_nodes_graph = self.graph.nodes()
        valid_combinations = []
        print("\t...filtering")

        for combination in self.all_combinations:
            if combination:
                if power_nodes_discretionary!=None:
                    if self.remove_trees_where_weak_nodes_are_hubs(weak_nodes, combination):
                        if debug:
                            print("\tcheck C")
                        if self.are_discretionary_nodes_singularly_connected(combination, power_nodes_discretionary):
                            if debug:
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

    def remove_trees_where_discretionary_not_connected_to_discretionary_or_mandatory(self, edges, power_nodes_mandatory, power_nodes_discretionary):
        graph = {}
        # Build the graph represented as an adjacency dictionary
        for edge in edges:
            u, v = edge
            if u not in graph:
                graph[u] = []
            if v not in graph:
                graph[v] = []
            graph[u].append(v)
            graph[v].append(u)
        
        # Recursive depth-first search function
        def dfs(graph, node, visited):
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(graph, neighbor, visited)
        
        # Check connectivity between mandatory and discretionary nodes
        visited_nodes = set()
        for node in power_nodes_mandatory:
            if node not in visited_nodes:
                dfs(graph, node, visited_nodes)
    
        return all(node in visited_nodes for node in power_nodes_discretionary if node in graph)


def create_graph( weak_nodes=None, power_nodes_mandatory=None, power_nodes_discretionary=None):
    G = nx.Graph()
    all_nodes=[]
    
    if (weak_nodes is not None) and (power_nodes_mandatory is not None) and (power_nodes_discretionary is not None):
        G.add_nodes_from(weak_nodes, node_type='weak')
        G.add_nodes_from(power_nodes_mandatory, node_type='power_mandatory')
        G.add_nodes_from(power_nodes_discretionary, node_type='power_discretionary')
        all_nodes = list(weak_nodes) + list(power_nodes_mandatory) + list(power_nodes_discretionary)
    elif (weak_nodes is not None) and (power_nodes_mandatory is not None):
        G.add_nodes_from(weak_nodes, node_type='weak')
        G.add_nodes_from(power_nodes_mandatory, node_type='power_mandatory')
        all_nodes = list(weak_nodes) + list(power_nodes_mandatory)
    elif (weak_nodes is not None) and (power_nodes_discretionary is not  None):
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
                # Assign a random weight between 1 and 10 (you can customize the range)
                weight = random.randint(1, 10)
                G.add_edge(i, j, weight=weight)
    return G


def draw_graph(G):
    plt.clf()
    global count_picture
    if debug2:
        print("\tnodes_", G.nodes(), " edges:", G.edges)
    pos = nx.spring_layout(G)
    node_colors = {'weak': 'green', 'power_mandatory': 'red', 'power_discretionary': 'orange'}
    colors = [node_colors[data['node_type']] for _, data in G.nodes(data=True)]
    edge_labels = {(i, j): G[i][j]['weight'] for i, j in G.edges()}
    
    nx.draw(G, pos, with_labels=True, node_color=colors, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()


count_picture=0
def save_graph(G, name=None):
    global count_picture
  
    if name==None or name=="best_tree":
        pos = nx.spring_layout(G)
        node_colors = {'weak': 'green', 'power_mandatory': 'red', 'power_discretionary': 'orange'}
        colors = [node_colors[data['node_type']] for _, data in G.nodes(data=True)]

        edge_labels = {(i, j): G[i][j]['weight'] for i, j in G.edges()}

        nx.draw(G, pos, with_labels=True, node_color=colors, font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    #plt.show()
    path_to_save = path+str(count_picture)+"_graph"+".png"

    if name!=None:
        path_to_save=path+str(count_picture)+"_"+str(name)+"_graph"+".png"
    
    # Save plot
    plt.savefig(path_to_save)
    plt.close()
    count_picture+=1


def draw_tree_highlighting_edges(G, list_edges, save=None):
    pos = nx.spring_layout(G)
    node_colors = {'weak': 'green', 'power_mandatory': 'red', 'power_discretionary': 'orange'}
    colors = [node_colors[data['node_type']] for _, data in G.nodes(data=True)]
    edge_labels = {(i, j): G[i][j].get('weight', None) for i, j in G.edges()}
    
    plt.clf() 
    nx.draw(G, pos, with_labels=True, node_color=colors, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw_networkx_edges(G, pos, edgelist=list_edges, edge_color='blue', width=2)
    
    if save==None or save==False:
        plt.show()
    else:
        save_graph(G, "colored_intermediate_graph")


added_edges=set()
def join_2_trees(graph1, graph2, weak_nodes=None, power_nodes_mandatory=None, power_nodes_discretionary=None):
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
                if (graph1.has_edge(i, j) or graph1.has_edge(j, i)):
                    G.add_edge(i, j, weight=graph1[i][j]['weight'])
                elif (graph2.has_edge(i, j) or graph2.has_edge(j,i)):
                    G.add_edge(i, j, weight=graph2[i][j]['weight'])
                elif (i,j) in added_edges or  (j,i) in added_edges:
                    G.add_edge(i, j, weight=added_edges[i][j]['weight'])
                elif matching_tuple:
                    weight_value = next(iter(matching_tuple[2]))  # Estrai il valore dal set 'weight'
                    G.add_edge(i, j, weight=weight_value)
                else:
                    print(i, j,  "not found")
                    # random weight
                    weight = random.randint(1, 10)
                    #weight = random.randint(1, 2)
                    G.add_edge(i, j, weight=weight)
                    new_element = (i, j, frozenset({weight}))
                    added_edges.add(new_element)
    return G


def generate_combinations(elements):
    number_of_nodes=len(elements)
    for x in range(1, number_of_nodes+1):
        for combo in combinations(elements, x):
            yield combo


def generate_combinations2(elements, r):
    for combo in combinations(elements, r):
        yield combo


def generate_combinations2_mod(elements, r):
    for combo in combinations(elements, r):
        if debug:
            print(combo)
        #input("----")
    
        if combo[0] in weak_nodes and combo[1] in weak_nodes:
            pass
        else:
            yield combo


def generate_graphs(graph, power_nodes_discretionary):
    print("\n2nd graph")
    graph2 = create_graph(power_nodes_discretionary=power_nodes_discretionary)
    if debug0:
        draw_graph(graph2)
    combinations_only_power_nodes_discretionary = generate_combinations(graph2.nodes)
    list_graph_with_discretionary=[]
    count=0
    # Stampa tutte le combinazioni
    for combo in combinations_only_power_nodes_discretionary:
        if combo:
            lista_risultante = []
            # Estrai gli elementi dalle coppie e aggiungili direttamente alla lista
            for coppia in combo:
                try:
                    lista_risultante.extend(coppia)
                except:
                    lista_risultante.append(coppia)

            if count==0:
                graph3=join_2_trees(graph, graph2, weak_nodes=weak_nodes, power_nodes_mandatory=power_nodes_mandatory, power_nodes_discretionary=lista_risultante)
                graph3_bak=graph3
                #draw_graph(graph3)
            else:
                graph3=join_2_trees(graph3_bak, graph2, weak_nodes=weak_nodes, power_nodes_mandatory=power_nodes_mandatory, power_nodes_discretionary=lista_risultante)
                #draw_graph(graph3)

            count+=1

            #list_graph_with_discretionary.append(graph3)
            print("\n\tgraph3_provv")
            save_graph(graph3)
            
            yield graph3
        
    
def get_weight(item):
    return item[1]['weight']


def compare_2_trees(tree1, tree2, power_nodes_mandatory, power_nodes_discretionary, capacities):
    if tree1==None:
        tree1=tree2

    print("\t\tcompare_2_trees")

    number_nodes_tree1=len(tree1.nodes())
    number_nodes_tree2=len(tree2.nodes())

    edges_with_weights1 = [(edge, tree1.get_edge_data(edge[0], edge[1])) for edge in tree1.edges()]
    max_edge_cost1=max(edges_with_weights1, key=get_weight)
    max_edge_cost1=max_edge_cost1[1]["weight"]

    edgecost1=0

    edges_with_weights2 = [(edge, tree2.get_edge_data(edge[0], edge[1])) for edge in tree2.edges()]
    max_edge_cost2=max(edges_with_weights2, key=get_weight)
    max_edge_cost2=max_edge_cost2[1]["weight"]
    
    edgecost2=0

    set_power_nodes=set(list(power_nodes_mandatory)+list(power_nodes_discretionary))
    cost_degree1=0
    for x in tree1.nodes():
        if x in set_power_nodes:
            try:
                tree1.degree(x)
                cost_degree1+=tree1.degree(x)/capacities[x]
            except AttributeError:
                print("error")

    cost_degree2=0
    for x in tree2.nodes():
        if x in set_power_nodes:
            try:
                tree2.degree(x)
                cost_degree2+=tree2.degree(x)/capacities[x]
            except AttributeError:
                print("error")
        
    if debug:
        print("cost_degree_:", cost_degree1, cost_degree2)
    if debug:
        print("cost_degree:", cost_degree1/len(tree1.nodes()), cost_degree2/len(tree2.nodes()))
    cost_degree1=cost_degree1/len(tree1.nodes())
    cost_degree2=cost_degree2/len(tree2.nodes())

    
    
  

    if max_edge_cost1>=max_edge_cost2:
        if debug:
            print("\tcase max1>max2, len:", len(edges_with_weights1), " :", edges_with_weights1)
        for edge1, data1 in edges_with_weights1:
            edgecost1+=data1['weight']
        if debug:
            print("\t\t-edgecost1:", edgecost1, " max1:", max_edge_cost1, " numbernodes1:", number_nodes_tree1)
        edgecost1=edgecost1/(max_edge_cost1*(len(edges_with_weights1)))


        for edge2, data2 in edges_with_weights2:
            edgecost2+=data2['weight']
        if debug:
            print("\t\t-edgecost2:", edgecost2, " max1:", max_edge_cost1, " numbernodes2:", number_nodes_tree2)
        edgecost2=edgecost2/(max_edge_cost1*(len(edges_with_weights1)))
          
    else:
        if debug:
            print("\tcase max2>max1, len:", len(edges_with_weights2), " :", edges_with_weights2)
        for edge1, data1 in edges_with_weights1:
            edgecost1+=data1['weight']
        if debug:
            print("\t\t-edgecost1:", edgecost1, " max1", max_edge_cost1, " numbernodes1:", number_nodes_tree1)
        edgecost1=edgecost1/(max_edge_cost2*(len(edges_with_weights2)))
      

        for edge2, data2 in edges_with_weights2:
            edgecost2+=data2['weight']
        if debug:
            print("\t\t-edgecost2:", edgecost2, " max2", max_edge_cost2, " numbernodes2:", number_nodes_tree2)
        edgecost2=edgecost2/(max_edge_cost2*(len(edges_with_weights2)))
        
    if debug:
        print("\nedge:", edgecost1, edgecost2, "\ndegree:", cost_degree1, cost_degree2, "\nsum:",edgecost1+cost_degree1, edgecost2+cost_degree2)

    #print("\t\tcost_degree_:", cost_degree1, cost_degree2)
    #print("\t\tedgecost1:", edgecost1, " edgecost2:", edgecost2)
            

    if edgecost1+cost_degree1<=edgecost2+cost_degree2:
        if debug:
            print("\n\n\nbest_tree")
        #return tree1, edgecost1/number_nodes_tree1, cost_degree1
        return tree1, edgecost1, cost_degree1

    else:
        if debug:
            print("\n\n\nnew_tree")
        #return tree2, edgecost2/number_nodes_tree2, cost_degree2
        return tree2, edgecost2, cost_degree2


def build_tree_from_list_edges(G, desired_edges, no_plot=None):
    G_copy=copy.deepcopy(G)
    edges_to_remove = [edge for edge in G_copy.edges() if edge not in desired_edges]
    G_copy.remove_edges_from(edges_to_remove)

    if no_plot==False or no_plot==None:
        pos = nx.spring_layout(G_copy)
        node_colors = {'weak': 'green', 'power_mandatory': 'red', 'power_discretionary': 'orange'}
        colors = [node_colors[data['node_type']] for _, data in G_copy.nodes(data=True)]
        edge_labels = {(i, j): G_copy[i][j].get('weight', None) for i, j in G_copy.edges()}

        nx.draw(G_copy, pos, with_labels=True, node_color=colors, font_weight='bold')
        nx.draw_networkx_edge_labels(G_copy, pos, edge_labels=edge_labels)
        nx.draw_networkx_edges(G_copy, pos, edgelist=desired_edges)
        plt.show()
    return G_copy
    

def process_graph(graph, weak_nodes, power_nodes_mandatory, power_nodes_discretionary, best_tree):
    if debug_plot_graph:
        draw_graph(graph)

    combinations_graph_with_some_discretionary = CombinationGraph(graph)
    if debug:
        input("\n\nENTER to continue...")

    combinations_graph_with_some_discretionary.filter_combinations_discretionary(weak_nodes, power_nodes_mandatory, power_nodes_discretionary)
    print("here4:", len(combinations_graph_with_some_discretionary.all_combinations))
    if debug:
        input("\n\nENTER to continue...")

  
    for x in combinations_graph_with_some_discretionary.all_combinations:
        if debug:
            print("nodes: ", x)
            print("\n\n33:", type(graph))
        if debug_plot_graph:
            draw_tree_highlighting_edges(graph, x)

        if debug_save:
            draw_tree_highlighting_edges(graph, x, save=True)

        if debug_plot_graph==False:
            tree=build_tree_from_list_edges(graph, x, no_plot=True)
        else:
            tree=build_tree_from_list_edges(graph, x, no_plot=False)

        best_tree, edgecost_best, degree_best=compare_2_trees(best_tree, tree, power_nodes_mandatory, power_nodes_discretionary, capacities)

        if debug:
            input("\n\nENTER to continue...")
            print("\tplot_best_tree")
        if debug_plot_graph:
            draw_graph(best_tree)

    return best_tree


#verify if the graph is connected
def dfs_modified(graph, start):
    visited = set()  # Set to keep track of visited nodes
    stack = [start]  # Stack to keep track of nodes to visit
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            # Find all neighbors adjacent to the current node
            neighbours = [neighbour for edge in graph for neighbour in edge if node in edge and neighbour != node]
            stack.extend(neighbour for neighbour in neighbours if neighbour not in visited)
    return visited


def is_connected(graph, number_of_nodes):
    start_node = graph[0][0]  # Take the first node as starting point
    reachable_nodes = dfs_modified(graph, start_node)
    # If the number of reachable nodes equals the total number of nodes in the graph, then the graph is connected
    return len(reachable_nodes) == number_of_nodes








if __name__ == "__main__":
    global check_only_discretionary #if num_nodes_discretionary==len(graph.nodes)
    global check_no_mandatory #if num_nodes_mandatory==0
    check_only_discretionary=False
    check_no_mandatory=False
    start_time = time.time()

    num_nodes = 8

    #critical cases: 0 weak, 0 mandatory, 0 discretionary, 0 weak e 0 mandatory, 0weak e 0 discretinary, 0 discretionary e 0 mandatory
    num_weak_nodes = int(0.4 * num_nodes)
    num_power_nodes_mandatory = int(0.2 * num_nodes)
    num_power_nodes_discretionary = num_nodes - num_weak_nodes - num_power_nodes_mandatory

    if num_weak_nodes==num_nodes:
        print("only weak nodes")
        sys.exit()
    elif num_power_nodes_discretionary==num_nodes:
        print("only discretionary")
        check_only_discretionary = True
        #sys.exit()
    if num_power_nodes_mandatory==0 or num_power_nodes_mandatory==1:
        check_no_mandatory=True

    weak_nodes = range(1, num_weak_nodes + 1)
    power_nodes_mandatory = range(num_weak_nodes + 1, num_weak_nodes + num_power_nodes_mandatory + 1)
    power_nodes_discretionary = range(num_weak_nodes + num_power_nodes_mandatory + 1, num_weak_nodes + num_power_nodes_mandatory + num_power_nodes_discretionary + 1)
    
    capacities = {1: 10, 2: 30, 3:2, 4: 1, 5: 10, 6:4, 7:5, 8:5, 9:5, 10:5, 11:5, 12:5, 13:5, 14:5, 15:5, 16:5, 17:5, 18:5, 19:5, 20:5}
    
    print("num_weak_nodes+num_power_nodes_mandatory:", num_weak_nodes+num_power_nodes_mandatory)
   
    if num_weak_nodes+num_power_nodes_mandatory>0:
        #Graph with power mandatory and weak nodes
        graph = create_graph(weak_nodes, power_nodes_mandatory, power_nodes_discretionary=None)
        print("1st graph")
        
        if debug0:
            draw_graph(graph)
        save_graph(graph)

        #Generating all combinations involving weak nodes and mandatory power nodes
        combinations_graph_with_some_discretionary2 = CombinationGraph(graph)
        #filtering
        combinations_graph_with_some_discretionary2.filter_combinations_discretionary(weak_nodes, power_nodes_mandatory)
        print("\there3:", len(combinations_graph_with_some_discretionary2.all_combinations))
        if debug:
            input("\n\nENTER to continue...")

        if len(combinations_graph_with_some_discretionary2.all_combinations) > 0:
            best_tree = build_tree_from_list_edges(graph, combinations_graph_with_some_discretionary2.all_combinations[0], no_plot=True)
        else:
            print("empty list")
            best_tree=None


        for x in combinations_graph_with_some_discretionary2.all_combinations:
            if debug:
                print("nodes: ", x)
                print("22:", type(graph))
            if debug_plot_graph:
                draw_tree_highlighting_edges(graph, x)
            if debug_save:
                draw_tree_highlighting_edges(graph, x, "colored_intermediate_graph")

            if debug_plot_graph==False:
                tree=build_tree_from_list_edges(graph, x, no_plot=True)
            else:
                tree=build_tree_from_list_edges(graph, x, no_plot=False)

            if debug2:
                print("\n\n44:", type(best_tree), type(tree))

            best_tree, edgecost_best, degree_best=compare_2_trees(best_tree, tree, power_nodes_mandatory, power_nodes_discretionary, capacities)
            
            if debug:
                input("Enter...")
            if debug_plot_graph:
                print("plot_best_tree")
                draw_graph(best_tree)

        if debug2:
            input("end 1st graph")
    else:
        graph = nx.Graph()
        best_tree=None


    
    graphs=generate_graphs(graph, power_nodes_discretionary)
    
    for graph in graphs:
        best_tree=process_graph(graph, weak_nodes, power_nodes_mandatory, power_nodes_discretionary, best_tree)


    end_time = time.time()
    if debug0:
        print("\n\n\n",best_tree)
        draw_graph(best_tree)
        pass
    save_graph(best_tree, name="best_tree")
    
    # Time elapsed
    elapsed_time = end_time - start_time
    print(f"Running time: {elapsed_time} seconds")
         
