MAPF-ASP program solves the Multi-Agent Path Finding (MAPF) problem using Answer Set Programming (ASP).

##########################################################################################################

Requirements: Python 3.x and Clingo (ASP solver).

For Windows, the .exe file for the ASP solver Clingo is added in this package. No installation should be needed.

For Linux, it can be installed with the command:
    
    sudo apt install gringo


For further information about downloading Clingo:

    https://potassco.org/clingo/


##########################################################################################################

Running MAPF-ASP:

python3 MAPF-ASP.py [-h] [-o <output-file>] <input-file>

positional arguments:
  <input-file>          name of the input file

optional arguments:
  -h, --help            show this help message and exit
  -o <output-file>, --out-file <output-file>
                        name of the output file (default name:plan.txt)

##########################################################################################################

Input File format:

s <row_size> <column_size>
m <makespan>
a <agent_count> 
<a1_init_row> <a1_init_col> <a1_goal_row> <a1_goal_col>
...
<a_N_init_row> <a_N_init_col> <a_N_goal_row> <a_N_goal_col>
o <obstacle_count>
<obs1_row> <obs1_col>
...
<obs_M_row> <obs_M_col>

An example input file (input_example.txt) and an example output file (plan_example.txt) are included in the package.


