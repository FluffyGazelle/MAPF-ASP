% optimizations 

% find the plan length for each agent
planLength(A,L) :- #max{T: plan(A,T-1,Y), plan(A,T,X), goal(A,X), X!=Y}=L, agent(A).
%:- plan(A,T,X), planLength(A,L), L < T.

% the following will minimize first the max planlength, and then the sum of plan lengths. 
% if you want to minimize only max makespan or sum of plan lengths, comment the relevant rules.

% minimize the max makespan
maxPlanLength(M) :- #max{T: planLength(A,T)} = M.
:~ maxPlanLength(M). [M@5]

% minimize the plan length
:~ planLength(A,L). [L@3,A]

#show planLength/2.