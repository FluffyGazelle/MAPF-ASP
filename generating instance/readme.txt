First run creating_instance.py as:
python creating_instance.py map scenario makespan cluster_num agent_num agent_num

It creates new instance file named f'instance_{map_file}_agent_num={agent_num}_c={cluster}_m={makespan}'.

Then we run create_graph_from_instance.py as:
python create_graph_from_instance.py new_instance



Example:
python creating_instance.py den312d.map den312d-random-25.scen 150 10 50
python create_graph_from_instance.py instance_den312d.map_agent_num=50_c=10_m=150

