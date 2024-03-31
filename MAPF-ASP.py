import argparse
import subprocess as sp
import sys
import math

def parse_input_args():
    parser = argparse.ArgumentParser(prog='MAPF-ASP',
                                    description='MAPF solver')
    
    parser.add_argument("input", metavar="<input-file>", help='name of the input file')
    
    parser.add_argument("-o","--out-file", type=str, action="store", default="plan.txt",
                        dest='out_file', metavar="<output-file>",
                        help='name of the output file (default name:plan.txt)')
                        
    #parser.add_argument('-m', type=int, default=0, dest="mode", choices=[0,1],  help="input mode (0 or 1), default=0")                

    results = parser.parse_args()

    input_file = results.input
    out_file = results.out_file
    #inp_mode = results.mode
    inp_mode = 0
    return input_file, out_file, inp_mode


def parse_inp_file_default(input_file):

    with open(input_file,"r") as f:
        lines = f.readlines()
    
    grid_x = grid_y = makespan = num_agents = inits = goals = num_obs = obstacles = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if "#" in line: pass
        elif "s" in line:
            size = line.split()
            grid_x = int(size[1])
            grid_y = int(size[2])
        elif "m" in line:
            makespan = int(line.split()[1])
        elif "a" in line:
            if  grid_y == False:
                return "Incorrect grid size!"
            num_agents = int(line.split()[1])
            inits = ""
            goals = ""
            for n in range(1,num_agents+1):
                l = lines[i+n]
                if not all(x.isdigit() or x.isspace() for x in l):
                    return "Incorrect agent position!"
                l = l.split()
                if not len(l) == 4: 
                    return "Incorrect agent position!"
                #agent = l[0]
                init_x = int(l[0])
                init_y = int(l[1])
                init_pos = (init_x-1)*grid_y + init_y
                if init_pos > grid_x*grid_y or init_x > grid_x or init_y > grid_y:
                    # print(init_pos)
                    return "Incorrect initial position of agent "+ str(n)+"!"
                goal_x = int(l[2])
                goal_y = int(l[3])
                goal_pos = (goal_x-1)*grid_y + goal_y
                if goal_pos > grid_x*grid_y or goal_x > grid_x or goal_y > grid_y:
                    return "Incorrect goal position of agent "+ str(n)+"!"
                inits += "init(" + str(n) + "," + str(init_pos) + ")."
                goals += "goal(" + str(n) + "," + str(goal_pos) + ")."
            i += num_agents
        elif "o" in line:
            num_obs = int(line.split()[1])
            obstacles = ""
            for n in range(1,num_obs+1):
                l = lines[i+n]
                if not all(x.isdigit() or x.isspace() for x in l):
                    return "Incorrect obstacle position!"
                l = l.split()
                if not len(l) == 2:
                    return "Incorrect obstacle position!"
                obs_x = int(l[0])
                obs_y = int(l[1])
                obs_pos = (obs_x-1)*grid_y+ obs_y
                if obs_pos > grid_x*grid_y or obs_x > grid_x or obs_y > grid_y:
                    return "Incorrect obstacle position!"
                obstacles += "obstacle(" + str(obs_pos) + ")."
            i += num_obs
        i+=1
    # print(inits)
    # print(goals)
    # print(obstacles)
    input_items = [grid_x,grid_y,makespan,num_agents,inits,goals, obstacles]
    if False in input_items:
        return "Missing item in input file!"
    return input_items

def generate_graph(rows,cols):
    edge_str = "vertex(1.." + str(rows*cols) + ").\n\n"
    
    for r in range(rows):
        for c in range(1,cols+1):
            cur_vertex = r*cols + c
            edges = []
            if c != 1:
                edges.append(cur_vertex-1)
            if c != cols:
                edges.append(cur_vertex+1)
            if r != 0:
                edges.append(cur_vertex-cols)
            if r != rows-1:
                edges.append(cur_vertex+cols)
            edges.sort()
            edge_str+="edge("+str(cur_vertex)+",("
            for v in edges:
                edge_str+=str(v)+";"
            edge_str = edge_str[:-1]+"))."

        edge_str += "\n"
    edge_str += "\n"
    return edge_str


def generate_instance(input_data):
    rows = input_data[0]
    print("input_data",input_data)
    cols = input_data[1]
    makespan_str = "#const t = " + str(input_data[2]) + ".\n"
    #time range time(0..t).

    graph_str = generate_graph(rows,cols)
    # print(graph_str)

    agents_str ="agent(1.." + str(input_data[3]) + ").\n\n"

    instance =  [makespan_str, "\ntime(0..t).\n", graph_str, agents_str, input_data[4]+"\n",input_data[5]+"\n\n", input_data[6]]

    with open("mapf_instance.lp","w+") as f:
        f.writelines(instance)

def run_clingo():
    cmd_list = ["clingo", "mapf_instance.lp", "mapf.lp", "optimizations.lp"]
    print("Solving MAPF...")
    sol= sp.Popen(cmd_list, stdout=sp.PIPE, stderr = sp.PIPE, encoding = 'utf8').communicate()[0]
    return sol

def parse_output(sol, num_agents):
    sol = sol.splitlines()[::-1]
    #print(sol)
    max_plan_length = total_plan_length = 0

    if "UNSATISFIABLE" in sol:
        return "UNSATISFIABLE", 0, []

    plans = [dict() for agent in range(num_agents)]
    planLengths = dict()

    cpu_found = opt_found = plan_found = False
    for line in sol:
        if "CPU" in line and not cpu_found:
            l = line.split()
            # print(l)
            cpu_time = l[-1]    
            cpu_found = True

        if "Optimization :" in line and not opt_found:
            l = line.split()
            # print(l)
            max_plan_length = l[2]
            total_plan_length = l[3]
            opt_found = True

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

                    plans[agent-1][time] = pos

                elif "planLength(" in item:
                    item = item.replace("planLength(","").replace(")","") 
                    parts = item.split(",")
                    agent = int(parts[0])
                    length = int(parts[1])
                    
                    planLengths[agent-1] = length
            print(planLengths)
            print(plans)
            plan_found = True

        if cpu_found and opt_found and plan_found:
            print(plans)
            return plans, planLengths, [max_plan_length, total_plan_length, cpu_time]

def write_to_out(plans, plan_lengths, out_file, grid_y, opts):
    out_list = []

    for agent in range(len(plans)):
        out_list.append("Agent "+str(agent+1)+":\n")

        plan = plans[agent]
        #plan_length = len(plan.values())
        plan_list = []

        for step in range(plan_lengths[agent]+1):
            x = math.ceil(plan[step]/grid_y)
            if plan[step]%grid_y == 0:
                y = grid_y
            else:
                y = plan[step]%grid_y
            plan_list.append("("+str(x)+","+str(y)+")")

        out_list.append("-".join(plan_list))
        out_list.append("\n\n")
    
    out_list.append("Maximum Plan Length: " + opts[0] + "\n")
    out_list.append("Total Plan Length: "+ opts[1] + "\n")
    out_list.append("CPU Time: "+ opts[2] + "\n")   
    
    with open(out_file,"w+") as f:
        f.writelines(out_list)


#######################

input_file, out_file, inp_mode = parse_input_args()

if inp_mode == 0:

    input_data = parse_inp_file_default(input_file)
    print("inputt",input_data)
    # print(input_data)
    if type(input_data) == list:
        # print(input_data)
        print("Generating instance...")
        rows = input_data[0]
        print("input_data",input_data)
        cols = input_data[1]
        makespan_str = "#const t = " + str(input_data[2]) + ".\n"
        #time range time(0..t).

        graph_str = generate_graph(rows,cols)

        with open("mapf_instance.lp","w+") as f:
            f.writelines(graph_str)
    else:
        print(input_data)

