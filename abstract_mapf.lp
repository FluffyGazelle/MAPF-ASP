% MAPF problem

% Given instance specifies time/1, edge/2, vertex/1, obstacle/1, agent/1, init/2, goal/2.

%%%%GENERATE%%%%

% for initial vertex X of agent A, there exists a path at time 0 
plan(A,0,X) :- init(A,X), agent(A).

% if there is a path to X at T-1 and an edge between X,Y ; a path to Y exists at time T.
1{plan(A,T,X); plan(A,T,Y):edge(X,Y)}1 :- plan(A,T-1,X), time(T).

%%%TEST%%%
% there exists an incoming edge to goal (to make sure the agent reaches its goal).
:- {plan(A,T,Y):edge(X,Y),goal(A,Y)}0, agent(A).

% if there exists a plan to X, an edge between X-Y and X is goal, there cannot be a plan to Y => there is no outgoing edge from goal.
:- plan(A,T,X), plan(A,T+1,Y), edge(X,Y), goal(A,X), agent(A), time(T).


%visit(A,X) :- plan(A,T,X).
%:- waypoint(A,X), not visit(A,X).



% in abstract graph, not more than C agents can visit a vertex at time T.
:- C{plan(A,T,X):agent(A)}, vertex(X), time(T), node_cap(X, C-1).

% Define movements from X to Y at time T
plan_to(X, Y, T, A) :- plan(A, T-1, X), plan(A, T, Y), edge(X, Y), time(T), agent(A).

:- C{ plan_to(X, Y, T, A) ; plan_to(Y, X, T, A) : agent(A) }, edge(X, Y), edge_cap(X, Y, C-1), time(T).


% Assign a unique door to each agent at each timestep   
1{ assigned_door(A, T-1, L) : door(edge(X, Y), L) } 1 :- agent(A), plan(A, T-1, X), plan(A, T, Y), time(T),X != Y. 

% display the resulting plans
#show plan/3.
#show assigned_door/3.
