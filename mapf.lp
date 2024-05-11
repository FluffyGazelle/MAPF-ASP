% MAPF problem

% Given instance specifies time/1, edge/2, vertex/1, obstacle/1, agent/1, init/2, goal/2.

%%%%GENERATE%%%%

% for initial vertex X of agent A, there exists a path at time 0 
plan(A,0,X) :- init(A,X), agent(A).

% if there is an edge from X to Y, there exists an edge from Y to X. graph is undirected.
edge(Y, X) :- edge(X, Y).

% if there is a path to X at T-1 and an edge between X,Y ; a path to Y exists at time T.
1{plan(A,T,X);plan(A,T,Y):edge(X,Y)}1 :- plan(A,T-1,X), time(T).

%%%TEST%%%
% there exists an incoming edge to goal (to make sure the agent reaches its goal).
:- {plan(A,T,Y):edge(X,Y),goal(A,Y)}0, agent(A).

% no two agents are at the same vertex at time T.
:- plan(A1,T,X), plan(A2,T,X), agent(A1;A2), A1<A2.

% if there is an obstacle at vertex X, no path can visit X.
:- plan(A,T,X), obstacle(X), agent(A), time(T).

% agent must be at its goal at the end of the plan.
:- not goal(A,X), plan(A,t,X), agent(A).

% no swapping
:- plan(A1,T,X), plan(A1,T+1,Y), plan(A2,T,Y), plan(A2,T+1,X), agent(A1;A2), A1<A2, time(T), edge(X,Y).

%visit(A,X) :- plan(A,T,X).
%:- waypoint(A,X), not visit(A,X).

% display the resulting plans
#show plan/3.

