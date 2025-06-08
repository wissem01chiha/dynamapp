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
import networkx as nx
from collimator.lazy_loader import LazyLoader

from sympy.core.function import AppliedUndef

if TYPE_CHECKING:
    import sympy as sp
else:
    sp = LazyLoader("sp", globals(), "sympy")


def draw_graph(G):
    e_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == 0]
    nx.draw(
        G,
        pos=nx.bipartite_layout(G, e_nodes, align="vertical"),
        with_labels=True,
        node_color="lightgray",
        font_size=8,
        font_color="black",
    )


def delete_V_nodes_with_zero_A(G, A, X):
    V_nodes_with_zero_A = [x for i, x in enumerate(X) if A[i] is not None]
    for node in V_nodes_with_zero_A:
        G.remove_node(str(node))


class IndexReduction:
    def __init__(self, t, eqs, knowns, notebook=False):
        self.t = t
        self.eqs = eqs
        self.knowns = knowns
        # Convert knowns to a set for efficient checking
        self.knowns_set = set(knowns)
        self.notebook = notebook
        self.initialize()

    def initialize(self):
        self.process_equations()
        self.create_bipartite_graph()
        self.create_association_list()
        self.assign = [None] * len(self.X)
        self.B = [None] * self.N
        self.Nprime = self.N
        self.Mprime = self.M

    def __call__(self):
        # self.initialize()
        self.pantelides()
        self.make_BLT_graph()
        self.dummy_derivatives()
        print("\n")
        print("#" * 10, "Final DAE equations", "#" * 10, "\n")
        for idx, eq in enumerate(self.final_dae_eqs):
            print(f"Eq {idx}:", eq)

        print("\n", "#" * 41)

    def process_equations(self):
        """
        f(x, ẋ, y ) = 0

        extract x, ẋ, and y from the list of eqs (f)
        """

        # Initialize empty sets for x, ẋ, and y
        x = set()
        x_dot = set()
        y = set()

        # Iterate through each equation
        for eq in self.eqs:
            # Find all functions and derivatives in the equation
            derivatives = eq.atoms(sp.Derivative)
            all_functions = eq.atoms(AppliedUndef)

            for derivative in derivatives:
                if (
                    derivative.args[0] not in self.knowns_set
                ):  # exclude derivatives of knowns
                    x_dot.add(derivative)
                    x.add(derivative.args[0])

            for func in all_functions:
                if func not in self.knowns_set:
                    y.add(func)

        # Remove vars from alg_vars, when derivative of var is in diff_vars
        y = y.difference(x)

        # All vars
        X = set().union(x, x_dot, y)

        self.x = list(x)
        self.x_dot = list(x_dot)
        self.y = list(y)
        self.X = list(X)

        self.n = len(x)
        self.m = len(y)

        self.N = self.n + self.m
        self.M = 2 * self.n + self.m

    def create_bipartite_graph(self):
        """
        Create a bipartite graph from the DAE system equations and variables.

        self.eqs (list): List of SymPy equations representing the DAE system.
        self.X (list): List of all variables (including derivatives) in the system.

        self.G: The created NetworkX bipartite graph.
        """

        G = nx.Graph()
        # Add nodes with the bipartite attribute
        G.add_nodes_from(
            [f"e{i}" for i, _ in enumerate(self.eqs)], bipartite=0
        )  # Equation nodes
        G.add_nodes_from([str(x) for x in self.X], bipartite=1)  # Variable nodes

        # Add edges based on variable presence in each equation
        for i, eq in enumerate(self.eqs):
            for term in self.X:
                if eq.has(term):
                    G.add_edge(f"e{i}", str(term))

        self.G = G

        self.e_nodes = [n for n, d in self.G.nodes(data=True) if d["bipartite"] == 0]
        self.v_nodes = [n for n, d in self.G.nodes(data=True) if d["bipartite"] == 1]

        # Create a mapping from node labels to indices and vice versa
        self.e_mapping = {node: idx for idx, node in enumerate(self.e_nodes)}
        self.v_mapping = {node: idx for idx, node in enumerate(self.v_nodes)}
        self.reverse_e_mapping = {idx: node for node, idx in self.e_mapping.items()}
        self.reverse_v_mapping = {idx: node for node, idx in self.v_mapping.items()}

        # Graph to keep track of equation differentiations
        self.eq_diff_graph = nx.DiGraph()
        self.eq_diff_graph.add_nodes_from([self.e_mapping[eq] for eq in self.e_nodes])

    def create_association_list(self):
        A = [None] * len(self.X)
        for idx, x in enumerate(self.X):
            if sp.diff(x, self.t) in self.X:
                A[idx] = self.X.index(sp.diff(x, self.t))
        self.A = A

    def augmentpath(self, G, i, pathfound, assign):
        e_node = self.reverse_e_mapping[i]
        # (1)
        nx.set_node_attributes(G, {e_node: {"color": "red"}})
        # (2)
        neighbors = list(G.neighbors(e_node))
        for v_node in neighbors:
            j = self.v_mapping[v_node]
            if assign[j] is None:
                pathfound = True
                assign[j] = i
                return pathfound, assign
        # (3)
        for v_node in neighbors:
            j = self.v_mapping[v_node]
            if G.nodes[v_node]["color"] == "white":
                nx.set_node_attributes(G, {v_node: {"color": "red"}})
                k = assign[j]
                pathfound, assign = self.augmentpath(G, k, pathfound, assign)
                if pathfound:
                    assign[j] = i
                    return pathfound, assign
        return pathfound, assign

    def pantelides(self, max_steps=100):
        # Step 3
        for k in range(self.Nprime):
            i = k
            pathfound = False
            counter_steps = 0
            while not pathfound and (counter_steps < max_steps):
                G = self.G.copy()
                delete_V_nodes_with_zero_A(G, self.A, self.X)
                nx.set_node_attributes(G, "white", "color")
                pathfound = False
                pathfound, self.assign = self.augmentpath(G, i, pathfound, self.assign)
                colored_e_nodes = [
                    n
                    for n, d in G.nodes(data=True)
                    if d["color"] == "red" and d["bipartite"] == 0
                ]
                colored_v_nodes = [
                    n
                    for n, d in G.nodes(data=True)
                    if d["color"] == "red" and d["bipartite"] == 1
                ]

                if not pathfound:
                    # (i)
                    for v_node in colored_v_nodes:
                        j = self.v_mapping[v_node]
                        self.M = self.M + 1

                        new_diff_var = sp.diff(self.X[j], self.t)
                        new_diff_var_name = str(new_diff_var)
                        self.X.append(new_diff_var)
                        self.G.add_node(new_diff_var_name, bipartite=1)
                        self.A.append(None)
                        self.assign.append(None)

                        self.v_mapping[new_diff_var_name] = (
                            self.M - 1
                        )  # -1 because of 0-based indexing
                        self.reverse_v_mapping[self.M - 1] = new_diff_var_name

                        self.A[j] = self.M - 1

                    # (ii)
                    for e_node in colored_e_nodes:
                        ll = self.e_mapping[e_node]
                        self.N = self.N + 1

                        new_eq_name = "e" + str(
                            self.N - 1
                        )  # -1 because of 0-based indexing
                        self.G.add_node(new_eq_name, bipartite=0)
                        self.B.append(None)
                        self.eqs.append(sp.diff(self.eqs[ll], self.t))

                        self.e_mapping[new_eq_name] = self.N - 1
                        self.reverse_e_mapping[self.N - 1] = new_eq_name

                        self.eq_diff_graph.add_node(self.N - 1)
                        self.eq_diff_graph.add_edge(ll, self.N - 1)

                        neighbors = self.G.neighbors(e_node)
                        for v_node in neighbors:
                            j = self.v_mapping[v_node]
                            self.G.add_edge(new_eq_name, v_node)
                            if self.A[j] is not None:
                                self.G.add_edge(
                                    new_eq_name, self.reverse_v_mapping[self.A[j]]
                                )

                        self.B[ll] = self.N - 1

                    # (iii)
                    for v_node in colored_v_nodes:
                        j = self.v_mapping[v_node]
                        self.assign[self.A[j]] = self.B[self.assign[j]]

                    # (iv)
                    i = self.B[i]
                counter_steps += 1

        assignment_dict = dict(
            zip(self.X, [f"e{i}" if i is not None else "" for i in self.assign])
        )
        eq_differentiation_dict = dict(
            zip(
                [f"e{i}" if i is not None else "" for i in range(len(self.B))],
                [f"e{i}" if i is not None else "" for i in self.B],
            )
        )

        derivative_mapping_dict = {
            self.X[base]: self.X[derivative]
            for base, derivative in enumerate(self.A)
            if derivative is not None
        }
        print("# Panteides Algorithm Completed", "\n")

        print(f"Total equations (before|after): {self.Nprime}|{self.N}")
        print(f"Total variables (before|after): {self.Mprime}|{self.M}")

        if self.notebook:
            print("\n")
            print("Variables", "\n")
            print(self.X)
            print("\n")
            print("Equations", "\n")
            for idx, eq in enumerate(self.eqs):
                print(eq)

        else:
            print("\n")
            print("Variables:", self.X)

            print("\n")
            print("Equations:", self.eqs)

        print("\n")
        print("# Variable assignments")
        for k, v in assignment_dict.items():
            print(f"Variable {k} is assigned to -> {v}")

        print("\n")
        print("# Differentiated equations")
        for k, v in eq_differentiation_dict.items():
            print(f"Differentiate {k} to get  -> {v}")

        print("\n")
        print("# Derivatives present in the variable association list")
        for k, v in derivative_mapping_dict.items():
            print(f"Present derivative of {k} is  -> {v}")

        # Process to keep only the assigned variables and equations
        self.pantelides_vars = [x for x, e in assignment_dict.items() if e]
        self.pantelides_eqs = [
            self.eqs[self.e_mapping[e]] for x, e in assignment_dict.items() if e
        ]
        self.pantelides_removed_eqs = list(set(self.eqs) - set(self.pantelides_eqs))

        # Variable to equation matching
        self.matching = {
            idx_var: idx_eq
            for idx_var, idx_eq in enumerate(self.assign)
            if idx_eq is not None
        }

        # Equation to variable matching
        self.reverse_matching = {
            idx_eq: idx_var
            for idx_var, idx_eq in enumerate(self.assign)
            if idx_eq is not None
        }

        self.index_to_var = {idx: var for idx, var in enumerate(self.X)}
        self.var_to_index = {var: idx for idx, var in enumerate(self.X)}

        self.p_dae_eqs = [
            eq_idx
            for eq_idx in self.eq_diff_graph.nodes()
            if self.eq_diff_graph.out_degree(eq_idx) == 0
        ]

        self.p_dae_vars = [self.reverse_matching[eq_idx] for eq_idx in self.p_dae_eqs]

        self.reverse_matching_p_dae = {
            eq_idx: self.reverse_matching[eq_idx] for eq_idx in self.p_dae_eqs
        }

    def find_vars_in_equation(self, eq):
        # Extract variables and their derivatives explicitly from the equation
        variables_and_derivatives = eq.atoms(sp.Derivative) | eq.atoms(sp.Function)

        # Define a function to check if a variable or its base function is a known symbol
        def is_known(var):
            if var in self.knowns_set:
                return True
            if isinstance(var, sp.Derivative):
                return var.expr in self.knowns_set
            return any(var.diff(t) in self.knowns_set for t in var.free_symbols)

        # Filter out known symbols and their derivatives
        variables_and_derivatives = {
            var for var in variables_and_derivatives if not is_known(var)
        }

        # Filter out base functions and lower-order derivatives if higher-order derivatives are present
        def filter_variables_and_derivatives(variables_set):
            filtered_set = set()
            for var in variables_set:
                if isinstance(var, sp.Derivative):
                    # Add higher-order derivatives directly
                    filtered_set.add(var)
                else:
                    # Check if the function or any of its derivatives are in the set
                    derivatives_present = any(
                        isinstance(v, sp.Derivative) and v.expr == var
                        for v in variables_set
                    )
                    if not derivatives_present:
                        filtered_set.add(var)
            return filtered_set

        return filter_variables_and_derivatives(variables_and_derivatives)

    # def find_vars_in_equation(self, eq):
    #     """
    #     Extract and filter variables and their derivatives from a given equation.
    #
    #     Parameters:
    #     eq (sympy expression): The equation from which to extract variables.
    #
    #     Returns:
    #     set: A set of filtered variables and derivatives present in the equation.
    #     """
    #     # Extract variables and their derivatives explicitly from the equation
    #     variables_and_derivatives = eq.atoms(sp.Derivative) | eq.atoms(sp.Function)
    #
    #     # Method to filter out base functions and lower-order derivatives if higher-order derivatives are present
    #     def filter_variables_and_derivatives(variables_set):
    #         filtered_set = set()
    #         for var in variables_set:
    #             if isinstance(var, sp.Derivative):
    #                 # Add higher-order derivatives directly
    #                 filtered_set.add(var)
    #             else:
    #                 # Check if the function or any of its derivatives are in the set
    #                 derivatives_present = any(
    #                     isinstance(v, sp.Derivative) and v.expr == var
    #                     for v in variables_set
    #                 )
    #                 if not derivatives_present:
    #                     filtered_set.add(var)
    #         return filtered_set
    #
    #     return filter_variables_and_derivatives(variables_and_derivatives)

    # def find_vars_in_equation(self, eq):
    #     """
    #     f(x, ẋ, y ) = 0
    #
    #     extract the variables present in the expression f(x, ẋ, y)
    #     """
    #
    #     # Extract variables and their derivatives explicitly
    #     variables_and_derivatives = eq.atoms(sp.Derivative) | eq.atoms(AppliedUndef)
    #
    #     # Filter out base functions if their derivatives are present
    #     filtered_variables = [
    #         var
    #         for var in variables_and_derivatives
    #         if isinstance(var, sp.Derivative)
    #         or all(
    #             var.diff(self.t) not in variables_and_derivatives
    #             for self.t in var.free_symbols
    #         )
    #     ]
    #
    #     return filtered_variables

    def make_BLT_graph(self):
        self.eq_idx_to_vars = {}
        self.eq_idx_to_vars_idx = {}
        for idx_eq, idx_matched_var in self.reverse_matching_p_dae.items():
            # Find all equations that contain the matched variable
            eq = self.eqs[idx_eq]
            all_vars_in_eq = self.find_vars_in_equation(eq)
            self.eq_idx_to_vars[idx_eq] = all_vars_in_eq
            all_vars_indices_in_eq = [self.X.index(var) for var in all_vars_in_eq]
            self.eq_idx_to_vars_idx[idx_eq] = all_vars_indices_in_eq

            print(f"equation e{idx_eq} is matched to variable v{idx_matched_var}")
            print(f"and contains the following variables: {all_vars_in_eq}")
            print(
                f"with indices: {all_vars_indices_in_eq}",
                "\n",
            )

        G = nx.DiGraph()
        G.add_nodes_from(self.reverse_matching_p_dae.keys())

        for idx_eq_parent, idx_matched_var in self.reverse_matching_p_dae.items():
            for idx_eq_child in self.reverse_matching_p_dae.keys():
                if idx_eq_child != idx_eq_parent:
                    if idx_matched_var in self.eq_idx_to_vars_idx[idx_eq_child]:
                        G.add_edge(idx_eq_parent, idx_eq_child)

        self.scc = [list(x) for x in nx.strongly_connected_components(G)]

        # Construct new graph with SCCs as nodes
        scc_graph = nx.DiGraph()
        scc_map = {}  # Map each node to its SCC
        for idx, component in enumerate(self.scc):
            scc_graph.add_node(idx)
            for node in component:
                scc_map[node] = idx

        # Add edges between SCCs in the new graph
        for u, v in G.edges():
            if scc_map[u] != scc_map[v]:
                scc_graph.add_edge(scc_map[u], scc_map[v])

        # Sort the SCCs in topological order
        topological_sorted_scc = list(nx.topological_sort(scc_graph))

        # Convert back to the actual nodes in the original graph
        self.BLT = [self.scc[idx] for idx in topological_sorted_scc]

        self.BLT_equation_ordering = [
            eq_idx for eq_set in self.BLT for eq_idx in eq_set
        ]
        self.BLT_variable_ordering = [
            self.reverse_matching_p_dae[eq_idx] for eq_idx in self.BLT_equation_ordering
        ]

        print("BLT equation ordering")
        print([self.reverse_e_mapping[idx] for idx in self.BLT_equation_ordering])

        print("BLT variable ordering")
        print([self.X[idx] for idx in self.BLT_variable_ordering])

        self.BLT_graph = G

    def sort_block_by_number_of_eq_derivatives(self, eq_block):
        G = self.eq_diff_graph

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

    def dummy_derivatives(self):
        BLT_eq_blocks = self.BLT
        self.dummy_vars = {}
        self.final_dae_eqs_pre_replacement = []
        self.replace = {}
        for unsorted_eq_block in BLT_eq_blocks:
            # Step 1
            num_parents, eq_block = self.sort_block_by_number_of_eq_derivatives(
                unsorted_eq_block
            )
            vars_block = [self.reverse_matching_p_dae[eq_idx] for eq_idx in eq_block]

            # j = 1

            g = sp.Matrix([self.eqs[eq_idx] for eq_idx in eq_block])
            z = sp.Matrix([self.X[var_idx] for var_idx in vars_block])
            G = g.jacobian(z)

            block_replace = {}
            sub_blocks = [eq_block]
            while True:
                # Step 2
                if sum(num_parents) == 0:
                    # Go to Step 6
                    break
                else:
                    # Step 3
                    m = sum([1 for n in num_parents if n != 0])

                    H = G[:m, :]

                    # Step 4
                    _, pivot_columns = H.rref()

                    # Step 5
                    M = H[:, pivot_columns]

                    for replacing_eq, replacing_var in zip(
                        eq_block[:m], [vars_block[idx] for idx in pivot_columns]
                    ):
                        block_replace[replacing_eq] = replacing_var

                    G = M
                    eq_block = [
                        list(self.eq_diff_graph.predecessors(eq_idx))[0]
                        for eq_idx in eq_block[:m]
                    ]
                    vars_block = [
                        self.A.index(vars_block[idx]) for idx in pivot_columns
                    ]

                    num_parents = [n - 1 for n in num_parents[:m]]
                    # j += 1

                    sub_blocks.append(eq_block)

            # Step 6
            # k = j

            final_block_eqs = []
            if block_replace:
                # Create dummy variables
                block_dummy_vars = {}
                for eq_idx, var_idx in block_replace.items():
                    dummy_var = sp.Symbol("d_" + str(self.X[var_idx]))
                    block_dummy_vars[self.X[var_idx]] = dummy_var
            else:
                block_dummy_vars = {}

            # Gather equations in reverse block order
            for sub_block in reversed(sub_blocks):
                for eq_idx in sub_block:
                    final_block_eqs.append(self.eqs[eq_idx])

            self.final_dae_eqs_pre_replacement.extend(final_block_eqs)
            self.dummy_vars.update(block_dummy_vars)
            self.replace.update(block_replace)

        # Replace true variables with dummy variables
        self.final_dae_eqs = []
        for eq in self.final_dae_eqs_pre_replacement:
            replaced_eq = eq.subs(self.dummy_vars)
            self.final_dae_eqs.append(replaced_eq)
