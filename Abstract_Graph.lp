#const t = 5.
time(0..t).
vertex(0).vertex(1).vertex(2).vertex(3).
edge(0, 1).edge(1, 0).edge(0, 3).edge(3, 0).edge(1, 2).edge(2, 1).edge(1, 3).edge(3, 1).edge(2, 3).edge(3, 2).
node_cap(0, 15).node_cap(1, 19).node_cap(2, 19).node_cap(3, 23).
edge_cap(0, 1, 4).edge_cap(0, 3, 4).edge_cap(1, 2, 4).edge_cap(1, 3, 1).edge_cap(2, 3, 4).
