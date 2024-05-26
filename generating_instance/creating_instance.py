import random
import sys

def create_instance():
    '''
    Takes 5 inputs in command line. Map file, Scenario file, Makespan, Number of Clusters, Number of agents.
    Creates intance file with the following format:
    s 4 5   // sizes height - width
    m 10    // makespan upper limit
    c 3     // number of cluster
    a 3     // number of agent
    1 5 4 2
    4 5 1 1
    3 1 1 4
    o 3     // number of obstacles
    3 5
    2 4
    2 2
    '''

    map_file = sys.argv[1]
    scen_file = sys.argv[2]
    makespan  = sys.argv[3]
    cluster = sys.argv[4]
    agent_num = int(sys.argv[5])


    # getting map size, and obstacles
    with open(map_file, 'r') as my_map_file:
        lines = my_map_file.readlines()


        height = int(lines[1].split()[1])
        width = int(lines[2].split()[1])

        line_num = 0
        obs_list = []
        for line in lines[4:]:
            for h in range(width):
                if line[h] != '.':
                    obs_list.append((line_num, h))
        
            line_num+= 1
    



    with open(scen_file, 'r') as scenario:

        # agent num


        lines = scenario.readlines()
     
        lines = lines[1:]
        random_agents = random.sample(lines, agent_num)


        agent_list = []
        for line in random_agents:
            agent_init_x = int(line.split()[4])
            agent_init_y = int(line.split()[5])
            agent_goal_x = int(line.split()[6])
            agent_goal_y = int(line.split()[7])
            agent_coord  = ((agent_init_x, agent_init_y), (agent_goal_x, agent_goal_y))
            agent_list.append(agent_coord)
        



    # return height, width, obs_list, agent_num, agent_list
    with open(f'instance_{map_file}_agent_num={agent_num}_c={cluster}_m={makespan}', 'w+') as instance:
        instance.write(f's {height}  {width} \n')
        instance.write(f'm {makespan} \n')
        instance.write(f'c {cluster} \n')
        instance.write(f'a {agent_num} \n')
        for agent in agent_list:
            instance.write(f'{agent[0][0]} {agent[0][1]} {agent[1][0]} {agent[1][1]}\n')
            # instance.write(f'{agent}\n')
        instance.write(f'o {len(obs_list)} \n')        
        for obs in obs_list:
            instance.write(f'{obs[0]} {obs[1]} \n')    
    







# height, width, obs_list, agent_num, agent_list = create_map_file_from_map_file(map_name, sce_name, 10, 4, 3)
# create_map_file_from_map_file(map_name, sce_name, 10, 4, 3)
create_instance()

# print("height: ", height)
# print("width: ", width)
# print("obs_list: ", obs_list)
# print("agent_num: ", agent_num)
# print("agent_list: ", agent_list)




