# Copyright (C) 2024 Collimator, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, version 3. This program is distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General
# Public License for more details.  You should have received a copy of the GNU
# Affero General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.

from typing import TYPE_CHECKING

from collimator.lazy_loader import LazyLoader

if TYPE_CHECKING:
    import networkx as nx
    from networkx.algorithms import bipartite
else:
    nx = LazyLoader("nx", globals(), "networkx")
    bipartite = LazyLoader("bipartite", globals(), "networkx.algorithms.bipartite")


def draw_bipartite_graph(G):
    e_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
    nx.draw(
        G,
        pos=nx.bipartite_layout(G, e_nodes, align="vertical"),
        with_labels=True,
        node_color="lightgray",
        font_size=8,
        font_color="black",
    )


def delete_var_nodes_with_zero_A(G, A, X):
    V_nodes_with_zero_A = [x for i, x in enumerate(X) if A[i] is not None]
    for node in V_nodes_with_zero_A:
        G.remove_node(node)


def augmentpath(G, i, pathfound, assign, v_mapping):
    """
    Algorithm 3.2 of
    Pantelides, C.C., 1988. The consistent initialization of differential-algebraic
    systems. SIAM Journal on scientific and statistical computing, 9(2), pp.213-231.
    """
    e_node = i
    # (1)
    nx.set_node_attributes(G, {e_node: {"color": "red"}})
    # (2)
    neighbors = list(G.neighbors(e_node))
    for v_node in neighbors:
        j = v_mapping[v_node]
        if assign[j] is None:
            pathfound = True
            assign[j] = i
            return pathfound, assign
    # (3)
    for v_node in neighbors:
        j = v_mapping[v_node]
        if G.nodes[v_node]["color"] == "white":
            nx.set_node_attributes(G, {v_node: {"color": "red"}})
            k = assign[j]
            pathfound, assign = augmentpath(G, k, pathfound, assign, v_mapping)
            if pathfound:
                assign[j] = i
                return pathfound, assign
    return pathfound, assign


def is_structurally_feasible(ics, Graph):
    G = Graph.copy()

    for ic_var in ics.keys():
        G.remove_node(ic_var)

    v_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 1]

    _mm = bipartite.matching.maximum_matching(G, top_nodes=v_nodes)
    mm = {k: v for k, v in _mm.items() if v in v_nodes}

    return len(mm) == len(v_nodes)


def sort_block_by_number_of_eq_derivatives(eq_diff_graph, eq_block):
    G = eq_diff_graph

    num_parents = []
    for eq_idx in eq_block:
        depth = 0
        current_node = eq_idx
        while G.in_degree(current_node) > 0:  # While the current node has a parent
            # Move to the parent node (only parent by design)
            current_node = list(G.predecessors(current_node))[0]
            depth += 1
        num_parents.append(depth)

    sorted_eqs = [
        eq_idx for _, eq_idx in sorted(zip(num_parents, eq_block), reverse=True)
    ]

    return sorted(num_parents, reverse=True), sorted_eqs


def tarjan_sort(G):
    """
    Perform Tarjan's algorithm to find and sort strongly connected components (SCCs)
    of a directed graph.

    Each SCC is returned as a list of nodes sorted by the reverse of their finishing
    times in the depth-first search.
    The list of SCCs is sorted in topological order, with the first SCC in the list
    being the last completed SCC in the depth-first search.

    Parameters:
    G (nx.DiGraph): A directed graph.

    Returns:
    List[List]: A list of SCCs, each represented by a list of nodes.
    """

    # Initialization
    index = 0
    indexes = {}
    lows = {}
    S = []  # Stack to maintain the order of visited nodes
    on_stack = set()  # Set to quickly check if a node is on the stack
    sorted_sccs = []  # List to hold the sorted strongly connected components

    def strongconnect(v):
        """
        Helper function to perform the depth-first search and identify SCCs.

        Parameters:
        v: The node to start the DFS from.
        """
        nonlocal index
        indexes[v] = index
        lows[v] = index
        index += 1
        S.append(v)
        on_stack.add(v)

        # Explore successors
        for w in G.successors(v):
            if w not in indexes:
                strongconnect(w)
                lows[v] = min(lows[v], lows[w])
            elif w in on_stack:
                lows[v] = min(lows[v], indexes[w])

        # If v is a root node, pop the stack and create an SCC
        if lows[v] == indexes[v]:
            scc = []
            while True:
                w = S.pop()
                on_stack.remove(w)
                scc.append(w)
                if w == v:
                    break
            # Reverse the SCC to maintain the order of finish times
            scc.reverse()
            sorted_sccs.append(scc)  # Append the SCC as a block

    # Process each node
    for v in G.nodes():
        if v not in indexes:
            strongconnect(v)

    # SCCs are completed in reverse topological order, so reverse the list
    sorted_sccs.reverse()
    return sorted_sccs
