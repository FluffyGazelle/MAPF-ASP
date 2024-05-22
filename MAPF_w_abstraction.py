import networkx as nx
import matplotlib.pyplot as plt
import pickle
import subprocess as sp
import sys
import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import os
import time
class Graphs():

    def parse_abstract_output(self, sol, num_agents):
        sol = sol.splitlines()[::-1]
        print(sol)
        

        if "UNSATISFIABLE" in sol:
            return "UNSATISFIABLE", 0, 0, 0

        plans = dict()
        planLengths = dict()
        doors = {}
        cpu_found = opt_found = plan_found = False
        for line in sol:
            if "CPU" in line and not cpu_found:
                l = line.split()
                print("cpu line",l)
                cpu_time = l[-1]    
                cpu_found = True
                
            if "plan" in line and not plan_found:
                # print(line)
                solution = line.split()
                #solution = line.replace("plan(","").replace(")","").split()
                #solution = [item.split(",") for item in solution]

                for item in solution:
                    if "plan(" in item:
                        item = item.replace("plan(","").replace(")","")
                        parts = item.split(",")
                        agent = int(parts[0])
                        time = int(parts[1])
                        pos = int(parts[2])
                        if agent not in plans:  # Initialize if not already present
                            plans[agent] = {}
                            doors[agent] = {}

                        plans[agent][time] = pos

                    elif "planLength(" in item:
                        item = item.replace("planLength(","").replace(")","") 
                        parts = item.split(",")
                        agent = int(parts[0])
                        if parts[1] == "#inf":
                            length = 0
                        else:
                            length = int(parts[1])
                        
                        planLengths[agent] = length
                    
                    elif "assigned_door(" in item:
                        item = item.replace("assigned_door(","").replace(")","")
                        parts = item.split(",")
                        print(parts)
                        agent = int(parts[0])
                        time = int(parts[1])
                        door = int(parts[2])
                        doors[agent][time] = door
                #print(planLengths)
                #print("plans", plans)
                plan_found = True
            
            if cpu_found and plan_found:
                #print(plans)
                print("plan lengths", planLengths)
                return plans, planLengths, doors, cpu_time
            
    def parse_output(self, sol, num_agents):
        sol = sol.splitlines()[::-1]
        print(sol)
        

        if "UNSATISFIABLE" in sol:
            return "UNSATISFIABLE", 0

        plans = dict()
        planLengths = dict()
        doors = {}
        cpu_found = opt_found = plan_found = False
        for line in sol:
            if "CPU" in line and not cpu_found:
                l = line.split()
                cpu_time = l[-1]    
                cpu_found = True

            if "plan" in line and not plan_found:
                # print(line)
                solution = line.split()
                #solution = line.replace("plan(","").replace(")","").split()
                #solution = [item.split(",") for item in solution]

                for item in solution:
                    if "plan(" in item:
                        item = item.replace("plan(","").replace(")","")
                        parts = item.split(",")
                        agent = int(parts[0])
                        time = int(parts[1])
                        pos = int(parts[2])
                        if agent not in plans:  # Initialize if not already present
                            plans[agent] = {}
                            doors[agent] = {}

                        plans[agent][time] = pos

                    
                #print(planLengths)
                #print("plans", plans)
                plan_found = True
            
            if cpu_found and plan_found:
                #print(plans)
                return plans, cpu_time
    
    def run_clingo_for_abstract_problem():
        cmd_list = ["clingo", "Abstract_Graph.lp", "Abstract_Agents.lp", "abstract_mapf.lp", "optimizations.lp"]
        print("Solving MAPF...")
        sol= sp.Popen(cmd_list, stdout=sp.PIPE, stderr = sp.PIPE, encoding = 'utf8').communicate()[0]
        return sol
    
    def run_clingo_for_subproblem(subgraph_file,subproblem_agents_file):
        print(subgraph_file)
        print(subproblem_agents_file)
        cmd_list = ["clingo", subgraph_file, subproblem_agents_file, "mapf.lp","subproblem_limit.lp"]
        print("Solving MAPF for subproblem...")
        sol= sp.Popen(cmd_list, stdout=sp.PIPE, stderr = sp.PIPE, encoding = 'utf8').communicate()[0]
        return sol
    
    def get_max_manhattan_distance(G):
        lowest = min(G.nodes(), key=lambda x: x[0] + x[1])
        highest = max(G.nodes(), key=lambda x: x[0] + x[1])

        # Calculate Manhattan distance
        manhattan_distance = abs(lowest[0] - highest[0]) + abs(lowest[1] - highest[1])

        return manhattan_distance
    """
    Problem 1: GRAPH PARTITIONING
    """
    def graph_partitioning(G, k):
        positions = {node: (node[0], node[1]) for node in G.nodes()}
        data_points = np.array(list(positions.values()))

        #Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data_points)

        labels = kmeans.labels_
        for i, node in enumerate(G.nodes()):
            G.nodes[node]['cluster'] = labels[i]
        
        
        """
        color_map = [f'C{G.nodes[node]["cluster"]}' for node in G.nodes()]
        plt.figure(figsize=(15, 15))
        nx.draw(G, pos=positions, node_color=color_map, with_labels=True, node_size= 100, font_weight='bold')
        plt.title('Graph Partitioning')
        plt.show()
        """
        subgraphs = []
        
        for cluster_id in range(k):
            
            cluster_subgraph = G.subgraph([node for node in G.nodes() if G.nodes[node]['cluster'] == cluster_id])
            cluster_subgraph.graph = cluster_subgraph.graph.copy()
            cluster_subgraph.graph['label'] = cluster_id
            subgraphs.append(cluster_subgraph)
        return subgraphs

    """
    Problem 2: SUB-GRAPH INTERSECTING
    """
    def subgraph_intersecting(self, G, subgraphs):
        modified_subgraphs = []

        # For every subgraph
        for i, H_i in enumerate(subgraphs):

            # Create a modified subgraph of H_i
            H_i_prime = nx.Graph(H_i)
            
             # in H_i, a node can be in only one clusters, but in H_i_prime, a node can be in multiple clusters so we need to convert the cluster value to a list
            for node in H_i_prime.nodes():
                cluster_val = H_i_prime.nodes[node].get('cluster', [])
                if not isinstance(cluster_val, list):
                    H_i_prime.nodes[node]['cluster'] = [cluster_val]
                    
            # Iterate over all edges in the original graph G
            for u, v in G.edges():
                # Check if one end of the edge is in H_i
                if u in H_i.nodes() or v in H_i.nodes():
                    # Find the subgraph index for the other end of the edge
                    for j, H_j in enumerate(subgraphs):
                        if j < i:
                            if (u in H_j.nodes() and v in H_i.nodes()) or (v in H_j.nodes() and u in H_i.nodes()):                                
                                
                                for H_j_prime in modified_subgraphs:
                                    sub_label = H_j_prime.graph['label'] if 'label' in H_j_prime.graph else None
                                    if H_j.graph['label'] == sub_label:
                                        break
                                
                                # Add both nodes and the edge to H_i_prime, if not already present
                                if not H_i_prime.has_node(u): 
                                    j_cluster_val = H_j.nodes[u]['cluster']
                                    i_cluster_val = H_i.nodes[v]['cluster']
                                    H_i_prime.add_node(u, cluster= [i_cluster_val, j_cluster_val])    #OKAY YOU F***ED UP HERE
                                    #print("added node", u)
                                    #print(H_i_prime.nodes[u]['cluster'])
                                    
                                if not H_i_prime.has_node(v): 
                                    j_cluster_val = H_j.nodes[v]['cluster']
                                    i_cluster_val = H_i.nodes[u]['cluster']

                                    H_i_prime.add_node(v,  cluster= [i_cluster_val, j_cluster_val])     #AAAAAAAA YOU F***ED UP HERE TOO
                                    #print("added node", v)
                                    #print(H_i_prime.nodes[v]['cluster'])

                                if not H_i_prime.has_edge(u, v): H_i_prime.add_edge(u, v)

            # Add the modified subgraph to the list
            modified_subgraphs.append(H_i_prime)

        return modified_subgraphs


    def add_cluster_to_node(self, H_i_prime, node, new_cluster):
        # Check if the node exists and has a 'cluster' attribute
        if node in H_i_prime and 'cluster' in H_i_prime.nodes[node]:
            # Append the new cluster value to the existing list
            H_i_prime.nodes[node]['cluster'].append(new_cluster)
        elif node in H_i_prime:
            # If the node doesn't have a 'cluster' attribute, create it
            H_i_prime.nodes[node]['cluster'] = [new_cluster]
    """
    Problem 3: ABSTRACT GRAPH REPRESENTATION
    """
    def abstract_graph_representation(modified_subgraphs, subgraph_colors):
        G_A = nx.Graph() #Abstract graph

        for i, H_i_prime in enumerate(modified_subgraphs):
            G_A.add_node(H_i_prime.graph['label'], size=len(H_i_prime.nodes()) - 1, color=subgraph_colors[i]) 
        node_sizes = nx.get_node_attributes(G_A, 'size')
        for H_i_prime in modified_subgraphs:
            for H_j_prime in modified_subgraphs:
                if H_i_prime.graph['label'] < H_j_prime.graph['label']:  
                    intersection = set(H_i_prime.nodes()) & set(H_j_prime.nodes())
                    print(H_j_prime.graph['label'], H_i_prime.graph['label'],"intersection", intersection)
                    if intersection:
                        G_A.add_edge(H_i_prime.graph['label'], H_j_prime.graph['label'], weight=len(intersection), doors = intersection) 
                    edge_weights = nx.get_edge_attributes(G_A, 'weight')
        return G_A, node_sizes, edge_weights

    """
    Problem 4: ABSTRACT AGENT POSITIONING
    """
    def abstract_agent_positioning(Agents, subgraphs):
        A_A = []  # Abstract agents set
        for agent in Agents:
            initial_pos, goal_pos = agent
            initial_subgraph_idx = None
            goal_subgraph_idx = None
            
            # Find which modified subgraph contains the initial and goal positions
            for i, H_i in enumerate(subgraphs):
                if initial_pos in H_i:
                    initial_subgraph_idx = i
                if goal_pos in H_i:
                    goal_subgraph_idx = i
            
            # Ensure both positions are found in the subgraphs
            if initial_subgraph_idx is not None and goal_subgraph_idx is not None:
                # Map to abstract graph nodes
                A_A.append(((initial_subgraph_idx, goal_subgraph_idx)))
        
        return A_A
    
    def subproblem_agents_to_str(A_s, obstacles):
        output_lines = []
        all_goals_in_subgraph = []
        # Write agents
        for agent in A_s.keys():
            output_lines.append(f"agent({agent}).")
        output_lines.append(f"\n")
        # Write initial positions and goals
        for agent, (initial_pos, goals) in A_s.items():
            # Convert initial position
            print("initial_pos for agent ", agent, initial_pos)
            print("initial and goal for agent ", agent, goals)
            print(H_i_prime.nodes())
            key = next((k for k, v in value_to_coord.items() if v == initial_pos), None)
            output_lines.append(f"init({agent},{key}).")
            
            # Convert each goal position
            for goal in goals:
                all_goals_in_subgraph.append(goal)
                output_lines.append(f"goal({agent},{goal}).")
            output_lines.append(f"\n")
            
        print("goals", goals)
        for obstacle in obstacles:
            print("the obs is ", obstacle)
            if obstacle not in all_goals_in_subgraph: 
                output_lines.append(f"obstacle({obstacle}).")
            
        return "".join(output_lines)


    def generate_lp_files(self, modified_subgraphs, G_A_with_colors, node_capacities, edge_capacities, A_A, value_to_coord, makespan_limit):
        for i, mod_sub in enumerate(modified_subgraphs):
            node_value_list = []    
            for node in mod_sub.nodes():
                key = next((k for k, v in value_to_coord.items() if v == node), None)
                node_value_list.append(key)
                
            formatted_string = ".".join([f"vertex({num})" for num in node_value_list])
            with open(f"subgraph_{mod_sub.graph['label']}.lp", "w") as file:
                file.write(formatted_string+".\n")
            
            edge_value_list = []    
            for node1, node2 in mod_sub.edges():
                key1 = [k for k, v in value_to_coord.items() if v == node1][0]

                key2 = [k for k, v in value_to_coord.items() if v == node2][0]

                edge_str = str(key1)+", "+str(key2)
                
                reverse_edge_str = str(key2)+", "+str(key1)
                edge_value_list.append(edge_str)
                edge_value_list.append(reverse_edge_str)
                formatted_string = ".".join([f"edge({num})" for num in edge_value_list])
            with open(f"subgraph_{mod_sub.graph['label']}.lp", "a") as file:
                file.write(formatted_string+".\n")
                
                
        abs_nodes = G_A_with_colors.nodes()
        abs_nodes_list = []
        for node in abs_nodes:
            abs_nodes_list.append(node)
        formatted_string = ".".join([f"vertex({num})" for num in abs_nodes_list])
        with open(f"Abstract_Graph.lp", "w") as file:
            file.write(f"#const t = {makespan_limit}.\ntime(0..t).\n")
            file.write(formatted_string+".\n")
            
        abs_edges = G_A_with_colors.edges(data=True)
        edge_value_list = []
        doors_formatted_string = ""
        formatted_string = ""
        for node1, node2, attr in abs_edges:
            edge_str = str(node1)+", "+str(node2)
            reverse_edge_str = str(node2)+", "+str(node1)

            edge_value_list.append(edge_str)
            edge_value_list.append(reverse_edge_str)
            
            doors = attr.get("doors")
            print("I am eager to learn wtf this is",edge_str, doors)
            abstract_graph_doors_list = list()
            for door in doors:
                key = next((k for k, v in value_to_coord.items() if v == door), None)
                abstract_graph_doors_list.append(key)
            
            
            input_str = f"door(edge({edge_str})"
            print("found it haha",input_str)
            doors_formatted_string += ".".join([f"{input_str},{num})" for num in abstract_graph_doors_list]) + ".\n"
            reverse_input_str = f"door(edge({reverse_edge_str})"
            doors_formatted_string += ".".join([f"{reverse_input_str},{num})" for num in abstract_graph_doors_list]) + ".\n"
            print("b",doors_formatted_string)
            
            formatted_string += ".".join([f"edge({num})" for num in edge_value_list[-2:]]) + "."
        
        with open(f"Abstract_Graph.lp", "a") as file:
            file.write(doors_formatted_string + formatted_string+"\n")
                
        formatted_string = ".".join([f"node_cap({key}, {value})" for key, value in node_capacities.items()])
        with open("Abstract_Graph.lp", "a") as file:
            file.write(formatted_string+".\n")
            
        formatted_string = ".".join([f"edge_cap({key[0]}, {key[1]}, {value})" for key, value in edge_capacities.items()])
        with open("Abstract_Graph.lp", "a") as file:
            file.write(formatted_string+".\n")
            
            
        formatted_string = ".".join([f"agent({num})" for num in range(len(A_A))])
        with open(f"Abstract_Agents.lp", "w") as file:
            file.write(formatted_string+".\n\n")
            for idx, (init, goal) in enumerate(A_A):
                
                file.write(f"init({idx},{init}).")
                
                file.write(f"goal({idx},{goal}).\n")

        

if __name__ == "__main__":
    start_time = time.time()

    fileName = "output" + ".txt"
    # create log folder
    folderName = "log"
    os.makedirs(folderName, exist_ok=True)
    # open file
    file_path = os.path.join(folderName, fileName)
    logFile = open(file_path, "w")
    sys.stdout = logFile
    sys.stderr = logFile
    #READ AGENTS FROM FILE
    print("Hi, starting the program...")
    total_makespan_limit = 34
    n = 2 # Number of clusters
    grid_size = (16, 16) 
    makespan_limit = n
    G = nx.grid_2d_graph(*grid_size)

    with open(f"{grid_size[0]}x{grid_size[1]}_agents/15_agents_1.txt", "r") as file:
        agents = file.read()
        agents_list = agents.split("\n")
        
    Agents = []
    for agent in agents_list:
        numbers_list = list(map(int, agent.split()))  # Convert each number to an integertegerr   
        paired_numbers = ((numbers_list[0], numbers_list[1]), (numbers_list[2], numbers_list[3]))
        Agents.append(paired_numbers) 
    print("agents:",Agents)       
    print("Generated the grid graph")
    
      

    positions = {node: (node[0], node[1]) for node in G.nodes()}

    value_to_coord = {(initx * max(grid_size)) + inity: (initx, inity) for initx, inity in G.nodes()}
    coord_to_value = {coord: value for value, coord in value_to_coord.items()}

    print("value to coord", coord_to_value)
    
    subgraphs = Graphs.graph_partitioning(G, n)

    modified_subgraphs = Graphs().subgraph_intersecting(G, subgraphs)
    subgraph_colors = [f'C{i}' for i in range(len(modified_subgraphs))]

    
    """
    for i, subgraph in enumerate(subgraphs):
        plt.figure(figsize=(5, 5))
        nx.draw(subgraph, pos=positions, with_labels=True, node_color=f'C{i}', font_weight='bold')
        plt.title(f'original Subgraph {i+1}')
    
    plt.show()
    

    for i, subgraph in enumerate(modified_subgraphs):
        plt.figure(figsize=(5, 5))
        nx.draw(subgraph, pos=positions, with_labels=True, node_color=f'C{i}', font_weight='bold')
        plt.title(f'Modified Subgraph {i+1}')

    plt.show()
"""
    
    # Generate the abstract graph with color
    G_A_with_colors, node_capacities , edge_capacities = Graphs.abstract_graph_representation(modified_subgraphs, subgraph_colors)

    A_A = Graphs.abstract_agent_positioning(Agents, subgraphs)
    print("abstract agents:", A_A)
    pos=nx.circular_layout(G_A_with_colors)
    pos=nx.spring_layout(G_A_with_colors,dim=2,pos=pos)

    """ 
    plt.figure(figsize=(10, 10))
    nx.draw(G_A_with_colors,
            with_labels=True,
            width=[G_A_with_colors[u][v]['weight'] for u, v in G_A_with_colors.edges()],
            edge_color='gray',
            node_color=[G_A_with_colors.nodes[node]['color'] for node in G_A_with_colors.nodes()],
            font_weight='bold')
    plt.title("Abstract Graph Representation with Subgraph Colors")
    plt.show()
    """
    
    
    # Generating the mapf instance files
    Graphs().generate_lp_files(modified_subgraphs, G_A_with_colors, node_capacities, edge_capacities, A_A, value_to_coord, makespan_limit)

    print("Running MAPF solver...")
    sol = Graphs.run_clingo_for_abstract_problem()
    print("sol",sol)
    solution, plan_lengths, doors_for_plans, abs_cpu_time = Graphs().parse_abstract_output(sol, len(A_A))
    print("doors",doors_for_plans)
    print(plan_lengths)
    print("solution",solution)
    if solution == "UNSATISFIABLE":
        print("No solution found for the given instance!")
        sys.exit(0)

    plans = {}
    
    longest_abstract_plan = max(plan_lengths.values())
    Agents_dict = {agent: Agents[agent] for agent in range(len(Agents))}
    PLANS = {}
    last_positions = {agent: Agents[agent][0] for agent in range(len(Agents))}
    
    subproblem_cpu_times = []
    curr_makespan = 0
    
    for abstract_step in range(longest_abstract_plan+1):
        print("\n\n\n\n\n\nabstract_step", abstract_step)
        print("Agents_dict", Agents_dict)
        print("last_positions", last_positions)
        makespan_for_step = 0
        all_doors_used_in_step  = {agent: doors[abstract_step] for agent, doors in doors_for_plans.items() if abstract_step in doors}
        cannot_be_goal_nodes = set(all_doors_used_in_step.values())
        print("cannot_be_goal_nodes", cannot_be_goal_nodes)
        planStep = []
        for H_i_prime in modified_subgraphs:
            #print("h prime nodes: ",H_i_prime.nodes())
            A_s = {}
            for H_i in subgraphs:
                sub_label = H_i.graph['label'] if 'label' in H_i.graph else None
                if H_i_prime.graph['label'] == sub_label:
                    break
            for agent_key in Agents_dict.keys():
                agent_value = Agents_dict[agent_key]
                print(solution)
                print("agent_solution", solution[agent_key], abstract_step)
                
                if solution[agent_key][abstract_step] == H_i_prime.graph['label']: #the agent is in the subgraph, add it to the subproblem containing the subgraph
                    print("subgraph label",H_i_prime.graph['label'])
                    
                    subproblem_goal_pos = []
                    print("agent", agent_key, agent_value)
                    subproblem_initial_pos = last_positions[agent_key]
                                 
                    if agent_value[1] in H_i_prime.nodes(): #the agent goal is in the subgraph
                        print("yay agent goal in subgraph",agent_value[1])
                        subproblem_goal_pos.append([k for k, v in value_to_coord.items() if v == agent_value[1]][0])
                    
                    else: #the agent goal is not in the subgraph
                        
                        if agent_key in doors_for_plans and abstract_step in doors_for_plans[agent_key]: #the door is assigned and the agent will move to another subgraph
                            subproblem_goal_pos.append(doors_for_plans[agent_key][abstract_step])
                            
                        else: #the door is not assigned and the agent will stay in the same subgraph
                            
                            filtered_nodes = [coord_to_value.get(node) for node in H_i.nodes() if coord_to_value.get(node) not in cannot_be_goal_nodes]
                            print("filtered nodes",filtered_nodes)
                            subproblem_goal_pos = subproblem_goal_pos + filtered_nodes                        

                        
                    init_goal_pos_for_agent_for_subproblem = (subproblem_initial_pos,subproblem_goal_pos)
                    print(init_goal_pos_for_agent_for_subproblem)
                    A_s[agent_key] = (init_goal_pos_for_agent_for_subproblem)

            obstacles = [coord_to_value.get(node) for node in H_i_prime.nodes() if coord_to_value.get(node) in cannot_be_goal_nodes]
            nodes_in_H_i_prime = set(H_i_prime.nodes())
            nodes_in_H_i = set(H_i.nodes())

            # Find the difference
            nodes_only_in_H_i_prime = nodes_in_H_i_prime - nodes_in_H_i
            print("nodes_in_H_i_prime",nodes_in_H_i_prime)
            print("nodes_in_H_i",nodes_in_H_i)
            # Convert the result to a list
            nodes_only_in_H_i_prime_list = list(nodes_only_in_H_i_prime)
            values_list = [coord_to_value[node] for node in nodes_only_in_H_i_prime if node in coord_to_value]
            combined_list = values_list + obstacles
            obstacles = list(set(combined_list))
            print("nodes_only_in_H_i_prime",nodes_only_in_H_i_prime)
            print("obs", values_list)
            print("obstacles", obstacles)
            print("A_s", A_s)
            if not A_s == {}:
                A_s_STR =Graphs.subproblem_agents_to_str(A_s, obstacles)
                output_path = f"subproblem_agents_for_step_{abstract_step}_subgraph_{H_i_prime.graph['label']}.lp"
                with open(output_path, 'w') as file:
                    file.write(A_s_STR)
                max_manhattan_distance = Graphs.get_max_manhattan_distance(H_i_prime)
                makespan_for_subproblem = int(max_manhattan_distance*0.7)
                while True:
                    with open("subproblem_limit.lp", "w") as file:
                        file.write(f"#const t = {makespan_for_subproblem}.\ntime(0..t).")
                    s_sol = Graphs.run_clingo_for_subproblem(f"subgraph_{H_i_prime.graph['label']}.lp", f"subproblem_agents_for_step_{abstract_step}_subgraph_{H_i_prime.graph['label']}.lp")
                    
                    print(s_sol)
                    s_solution, cpu_time_for_subproblem = Graphs().parse_output(s_sol, len(A_A))
                    print(s_solution)
                    if s_solution == "UNSATISFIABLE":
                        print("No solution found for the given instance!")
                        makespan_for_subproblem += 1
                        subproblem_cpu_times.append(cpu_time_for_subproblem)
                    else:
                        if makespan_for_step <= makespan_for_subproblem:
                            makespan_for_step = makespan_for_subproblem
                        break
                print("slution:", s_solution)
                last_elements = {key: value_to_coord[values[max(values.keys())]] for key, values in s_solution.items()}
                
                print("last_elements", last_elements)
                last_positions.update(last_elements)
                print("last positions",abstract_step, last_positions,"\n")
                planStep.append(s_solution)
        print("makespan_for_step",makespan_for_step)
        curr_makespan += makespan_for_step
        if curr_makespan > total_makespan_limit:
            print("Total makespan limit exceeded")
            print("No solution found for the given instance!")
            sys.exit(0)        
        max_size = max(len(inner_dict) for outer_dict in planStep for inner_dict in outer_dict.values())

        # Extend each inner dictionary to the maximum size with a default value (e.g., 0)
        for outer_dict in planStep:
            for key, inner_dict in outer_dict.items():
                for i in range(max_size):
                    if i not in inner_dict:
                        inner_dict[i] = inner_dict[i-1]
        print("planStep", planStep)
        abs_step_plans = {abstract_step: planStep}


        PLANS.update(abs_step_plans)
        print("PLANS for the abss tep",abstract_step, PLANS)
    print("longest_abstract_plan", longest_abstract_plan)

    print(PLANS)
    combined_plans = {}

    for timestep, agents_list in PLANS.items():
        for agents in agents_list:
            for agent_id, agent_plan in agents.items():
                if agent_id not in combined_plans:
                    combined_plans[agent_id] = []
                for step, init_value in agent_plan.items():
                    
                    combined_plans[agent_id].append(tuple(value_to_coord[init_value]))

    # ensuring chronological order of plans for each agent
    for agent_id in combined_plans:
        combined_plans[agent_id] = {i: step for i, step in enumerate(combined_plans[agent_id])}

    solution_valid = True
    longest_plan_agent = None
    max_length = 0
    #print("combined_plans", combined_plans)
    for agent_id, plan in combined_plans.items():
        print(f"\nAgent {agent_id} plan:")
        print(plan)
        if len(plan) > max_length:
            max_length = len(plan)
            longest_plan_agent = agent_id
        for agent_id_to_compare, plan_to_compare in combined_plans.items():
            if agent_id > agent_id_to_compare:
                for step, position in plan.items():
                    if step in plan_to_compare and plan_to_compare[step] == position:
                        solution_valid = False
                        print(f"Conflict at step {step} between agents {agent_id} and {agent_id_to_compare}")
    
    if solution_valid:
        print("Solution is valid!")
        
    end_time = time.time()

    # Calculate the total time taken
    elapsed_time = end_time - start_time
    total_subproblem_time = sum(float(t[:-1]) if isinstance(t, str) and t.endswith('s') else float(t) for t in subproblem_cpu_times)

    print(f"The script took {elapsed_time:.2f} seconds to run.")
    print(f"Abstract Problem solving time: {abs_cpu_time}")
    print("Amount of subproblems solved:", len(subproblem_cpu_times))
    print(f"Total subproblem solving time: {total_subproblem_time}s")
    print("max makespan", max_length)
