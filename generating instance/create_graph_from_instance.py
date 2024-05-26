import networkx as nx
import sys



def create_graph_from_instance():
    '''
    Creates a networkx graph from instance file.
    It returns height, width, makespan, number of cluster, agent list and graph itself in this order.

    '''    
    instance = sys.argv[1]
    with open(instance, 'r') as f:
        lines = f.readlines()


        height = int(lines[0].split()[1])
        width = int(lines[0].split()[2])
        makespan = int(lines[1].split()[1])
        cluster_num = int(lines[2].split()[1])
        agent_num = int(lines[3].split()[1])

        



        agent_list = []
        for agent_line in lines[4:4+agent_num]:
            # int(agent_line.split()[1])
            # agent = ((int(agent_line[0]),int(agent_line[1]) ), (int(agent_line[2]),int(agent_line[3]) ))
            agent = ((  int(agent_line.split()[0]) ,int(agent_line.split()[1])) , ( int(agent_line.split()[2]),int(agent_line.split()[3])))
            agent_list.append(agent)
        
        obst_line = 4+agent_num
        obstacle_num = int(lines[obst_line].split()[1])
        graph = nx.grid_2d_graph(height, width)
        for line in lines[obst_line+1:]:
            obs_h = int(line.split()[0])
            obs_w = int(line.split()[1])
            graph.remove_node((obs_h, obs_w))


    return height, width, makespan,cluster_num, agent_list, graph


# Example usage

ms, cl, agents, G = create_graph_from_instance()
# print("Nodes:", graph.nodes())


print(len(G.nodes()))
print("Nodes:", G.nodes())

print("\n\n\n edges \n")
print(G.edges())


print("Makespan:", ms)
print("Clusters:", cl)
# nx.draw(graph, with_labels=True, node_color='skyblue', node_size=100, edge_color='k', linewidths=1, font_size=1)
# plt.show()
