import networkx as nx
import numpy as np
import stim
from qiskit import QuantumCircuit

from pauliopt.pauli.clifford_gates import CliffordGate, CLiffordType, SingleQubitGate, \
    ControlGate
from pauliopt.topologies import Topology


def reconstruct_tableau(tableau: stim.Tableau):
    xsx, xsz, zsx, zsz, x_signs, z_signs = tableau.to_numpy()
    x_row = np.concatenate([xsx, xsz], axis=1)
    z_row = np.concatenate([zsx, zsz], axis=1)
    return np.concatenate([x_row, z_row], axis=0).astype(np.int64)


def reconstruct_tableau_signs(tableau: stim.Tableau):
    xsx, xsz, zsx, zsz, x_signs, z_signs = tableau.to_numpy()
    signs = np.concatenate([x_signs, z_signs]).astype(np.int64)
    return signs


def find_edges_up(G: nx.Graph, root: int):
    vs = {root}
    visited = {v: False for v in G.nodes}
    visited[root] = True
    steiner_up = []
    while vs:
        v = vs.pop()
        children = G.neighbors(v)
        for v_c in children:
            if not visited[v_c]:
                vs.add(v_c)
                visited[v_c] = True
                steiner_up.append((v, v_c))

    return list(reversed(steiner_up))


def compute_steiner_tree_up_down(root: int, pivot_root: int, nodes: [int],
                                 remaining_nodes: [int], topology: Topology):
    nodes_ = list(set([root, pivot_root] + nodes))
    resulting_edges_down = compute_steiner_tree(pivot_root, nodes_, remaining_nodes,
                                                topology)

    G = nx.Graph()
    G.add_edges_from(resulting_edges_down)

    if resulting_edges_down:
        resulting_edges_up = find_edges_up(G, root)
    else:
        resulting_edges_up = []

    return resulting_edges_down, resulting_edges_up


def get_unvisted_neighbours(T, leaf, visited):
    return list(filter(lambda x: not visited[x], T.neighbors(leaf)))


def produce_reduction_order(topo: Topology):
    reduction_order = []
    G = topo.to_nx
    T = nx.minimum_spanning_tree(G)
    assert isinstance(T, nx.Graph)

    leaf_node = \
        [node for node in range(topo.num_qubits) if len(list(T.neighbors(node))) == 1][0]
    queue = [leaf_node]
    visited = {node: False for node in range(topo.num_qubits)}
    visited[leaf_node] = True

    while queue:
        leaf = queue.pop(0)
        sorted_neighbours = sorted(get_unvisted_neighbours(T, leaf, visited),
                                   key=lambda x: len(
                                       get_unvisted_neighbours(T, x, visited)))
        assert visited[leaf]
        reduction_order.append(leaf)

        for neighbour in sorted_neighbours:
            visited[neighbour] = True
            queue.append(neighbour)
    return list(reversed(reduction_order))


def compute_steiner_tree(root: int, nodes: [int], remaining_nodes: [int],
                         topology: Topology):
    # TODO there must be an other more efficient way
    couplings = sorted([list(c.as_pair) for c in topology.couplings])
    t_dict = {
        "num_qubits": topology.num_qubits,
        "couplings": [(k, v) for k, v in couplings if
                      k in remaining_nodes and v in remaining_nodes]
    }
    topology_ = Topology.from_dict(t_dict)
    vertices = [root]
    nodes_steiner = list(set([root] + [n for n in nodes]))
    edges = []
    steiner_pnts = []
    while nodes_steiner:
        # find all connected options on the graph
        options = [(node, v, topology_.dist(int(node), int(v))) for node in nodes_steiner
                   for v in
                   (vertices + steiner_pnts) if
                   topology_.dist(int(node), int(v)) != np.inf]
        if not options:
            raise Exception("Topology is not connected or tableau is ill defined.")
        best_node, best_v, best_dist = min(options, key=lambda x: x[2])

        path = topology_.shortest_path(best_v, best_node)
        best_path = [(path[i - 1], path[i]) for i in range(1, len(path))]

        vertices.append(best_node)

        edges += best_path
        steiner = [v for edge in best_path for v in edge if v not in vertices]
        steiner_pnts += steiner
        nodes_steiner.remove(best_node)
    edges = list(set(edges))

    vs = {root}

    n_edges = len(edges)
    resulting_edges_down = []

    while len(resulting_edges_down) < n_edges:
        es = [e for e in edges for v in vs if e[0] == v]
        old_vs = [v for v in vs]
        for e1, e2 in es:
            resulting_edges_down.append((e1, e2))
            vs.add(e2)
        for v in old_vs:
            vs.remove(v)
    return resulting_edges_down


class CliffordTableau:
    def __init__(self, n_qubits: int = None, tableau: np.array = None,
                 signs: np.array = None):
        if n_qubits is None and tableau is None:
            raise Exception("Either Tableau or number of qubits must be defined")
        if tableau is None:
            self.tableau = np.eye(2 * n_qubits)
            self.signs = np.zeros((2 * n_qubits))
            self.n_qubits = n_qubits
        else:
            if tableau.shape[0] != tableau.shape[1]:
                raise Exception(
                    f"Must be a 2nx2n Tableau, but is of shape: {tableau.shape}")
            self.n_qubits = int(tableau.shape[1] / 2.0)
            self.tableau = tableau
            if signs is None:
                self.signs = np.zeros((2 * self.n_qubits))
            else:
                self.signs = signs
            if not 2 * self.n_qubits == tableau.shape[1]:
                raise Exception(
                    f"Tableau of shape: {tableau.shape}, is not a factor of 2")

    def apply_h(self, qubit):
        self.signs = (self.signs + self.tableau[:, qubit] * self.tableau[:,
                                                            self.n_qubits + qubit]) % 2

        self.tableau[:, [self.n_qubits + qubit, qubit]] = self.tableau[:,
                                                          [qubit, self.n_qubits + qubit]]

    def apply_s(self, qubit):
        self.signs = (self.signs + self.tableau[:, qubit] * self.tableau[:,
                                                            self.n_qubits + qubit]) % 2

        self.tableau[:, self.n_qubits + qubit] = (self.tableau[:, self.n_qubits + qubit] +
                                                  self.tableau[:, qubit]) % 2

    def apply_cnot(self, control, target):
        x_ia = self.tableau[:, control]
        x_ib = self.tableau[:, target]

        z_ia = self.tableau[:, self.n_qubits + control]
        z_ib = self.tableau[:, self.n_qubits + target]

        self.tableau[:, target] = \
            (self.tableau[:, target] + self.tableau[:, control]) % 2
        self.tableau[:, self.n_qubits + control] = \
            (self.tableau[:, self.n_qubits + control] + self.tableau[:,
                                                        self.n_qubits + target]) % 2

        tmp_sum = ((x_ib + z_ia) % 2 + np.ones(z_ia.shape)) % 2
        self.signs = (self.signs + x_ia * z_ib * tmp_sum) % 2

    def append_gate(self, gate: CliffordGate):
        if gate.c_type == CLiffordType.H:
            assert isinstance(gate, SingleQubitGate)
            self.apply_h(gate.qubit)
        elif gate.c_type == CLiffordType.S:
            assert isinstance(gate, SingleQubitGate)
            self.apply_s(gate.qubit)
        elif gate.c_type == CLiffordType.V:
            assert isinstance(gate, SingleQubitGate)
            self.apply_h(gate.qubit)
            self.apply_s(gate.qubit)
            self.apply_h(gate.qubit)
        elif gate.c_type == CLiffordType.CX:
            assert isinstance(gate, ControlGate)
            self.apply_cnot(gate.control, gate.target)
        elif gate.c_type == CLiffordType.CY:
            assert isinstance(gate, ControlGate)
            self.apply_s(gate.target)
            self.apply_s(gate.target)
            self.apply_s(gate.target)
            self.apply_cnot(gate.control, gate.target)
            self.apply_s(gate.target)
        elif gate.c_type == CLiffordType.CZ:
            assert isinstance(gate, ControlGate)
            self.apply_h(gate.target)
            self.apply_cnot(gate.control, gate.target)
            self.apply_h(gate.target)
        else:
            raise TypeError(
                f"Unrecongnized Gate type: {type(gate)} for Clifford Tableaus")

    @staticmethod
    def from_circuit(circ: QuantumCircuit):
        tableau = CliffordTableau(n_qubits=circ.num_qubits)
        for op in circ:
            if op.operation.name == "h":
                tableau.apply_h(op.qubits[0].index)
            elif op.operation.name == "s":
                tableau.apply_s(op.qubits[0].index)
            elif op.operation.name == "cx":
                tableau.apply_cnot(op.qubits[0].index, op.qubits[1].index)
        return tableau

    def inverse(self):

        x2x = self.tableau[:self.n_qubits, :self.n_qubits].copy().astype(np.bool8)
        z2z = self.tableau[self.n_qubits:2 * self.n_qubits,
              self.n_qubits:2 * self.n_qubits].copy().astype(np.bool8)
        x2z = self.tableau[:self.n_qubits, self.n_qubits:2 * self.n_qubits].copy().astype(
            np.bool8)
        z2x = self.tableau[self.n_qubits:2 * self.n_qubits, :self.n_qubits].copy().astype(
            np.bool8)
        tab = stim.Tableau.from_numpy(x2x=x2x,
                                      x2z=x2z,
                                      z2x=z2x,
                                      z2z=z2z,
                                      x_signs=self.signs[0:self.n_qubits].copy().astype(
                                          np.bool8),
                                      z_signs=self.signs[
                                              self.n_qubits:2 * self.n_qubits].copy().astype(
                                          np.bool8))
        t_inv = tab.inverse()
        ct_inv = reconstruct_tableau(t_inv)
        ct_signs = reconstruct_tableau_signs(t_inv)
        # x_row = np.concatenate([xsx, xsz], axis=1)
        # z_row = np.concatenate([zsx, zsz], axis=1)
        # inv_tableau = np.concatenate([x_row, z_row], axis=0).astype(np.int64)
        # p_string = np.zeros((2 * self.n_qubits))
        # p_string[1] = 1
        # print((inv_tableau @ (self.tableau @ ((self.signs * p_string) % 2) % 2)) % 2)

        return CliffordTableau(tableau=ct_inv, signs=ct_signs)

    def print_zx(self, X=True, Z=True):
        def x_out(inp, out):
            return self.tableau[inp, out] + \
                   2 * self.tableau[inp, out + self.n_qubits]

        def z_out(inp, out):
            return self.tableau[inp + self.n_qubits, out] + \
                   2 * self.tableau[inp + self.n_qubits, out + self.n_qubits]

        if X:
            print("X: ")
            for i in range(self.n_qubits):
                for j in range(self.n_qubits):
                    print(f" {int(x_out(i, j))} ", end="")
                print()
        if Z:
            print("Z: ")
            for i in range(self.n_qubits):
                for j in range(self.n_qubits):
                    print(f" {int(z_out(i, j))} ", end="")
                print()

    def to_cifford_circuit_arch_aware(self, topo: Topology):
        qc = QuantumCircuit(self.n_qubits)

        remaining = self.inverse()

        def apply(gate_name: str, gate_data: tuple):
            if gate_name == "CNOT":
                remaining.apply_cnot(gate_data[0], gate_data[1])
                qc.cx(gate_data[0], gate_data[1])
            elif gate_name == "H":
                remaining.apply_h(gate_data[0])
                qc.h(gate_data[0])
            elif gate_name == "S":
                remaining.apply_s(gate_data[0])
                qc.s(gate_data[0])
            else:
                raise Exception("Unknown Gate")

        def sanitize_field_z(row, column):
            if z_out(row, column) == 3:
                apply("S", (column,))

            if z_out(row, column) == 1:
                apply("H", (column,))

        def sanitize_field_x(row, column):
            if x_out(row, column) == 3:
                apply("S", (column,))

            if x_out(row, column) == 2:
                apply("H", (column,))

        def steiner_reduce_column(root, pivot_root, remaining_nodes):
            column_x = [row for row in remaining_nodes if x_out(root, row) != 0]
            for col in column_x:
                sanitize_field_x(root, col)
            resulting_down_x, resulting_up_x = compute_steiner_tree_up_down(root,
                                                                            pivot_root,
                                                                            column_x,
                                                                            remaining_nodes,
                                                                            topo)
            for row, col in resulting_down_x:
                assert (x_out(root, row) == 1 or x_out(root, row) == 0) and \
                       (x_out(root, col) == 1 or x_out(root, col) == 0)

                # This may cancel a zero, we will recover it (if necessary by first checking Z on the upward path
                if x_out(root, col) == 0 and x_out(root, row) == 1:
                    apply("CNOT", (row, col))

            for row, col in resulting_up_x:
                assert (x_out(root, row) == 1 or x_out(root, row) == 0) and \
                       (x_out(root, col) == 1 or x_out(root, col) == 0)
                if x_out(root, col) == 1 and x_out(root, row) == 1:
                    apply("CNOT", (row, col))

            # When the pivot is a zero in the column, pick a new one to cancel the Z Basis
            # TODO check what performs better
            # if z_out(root, pivot_root) == 0:
            pivot_root = pick_pivot_row(pivot, remaining_nodes)

            # sanatize all column elements, check that x is not 3
            column_z = [row for row in remaining_nodes if z_out(root, row) != 0]
            for col in column_z:
                sanitize_field_z(root, col)
            if x_out(root, root) != 1:
                apply("S", (root,))

            resulting_down_z, resulting_up_z = compute_steiner_tree_up_down(root,
                                                                            pivot_root,
                                                                            column_z,
                                                                            remaining_nodes,
                                                                            topo)
            for row, col in resulting_down_z:

                assert (z_out(root, row) == 2 or z_out(root, row) == 0) and \
                       (z_out(root, col) == 2 or z_out(root, col) == 0)

                if z_out(root, col) == 0 and z_out(root, row) == 2:
                    apply("CNOT", (col, row))

            for row, col in resulting_up_z:
                assert (z_out(root, row) == 2 or z_out(root, row) == 0) and \
                       (z_out(root, col) == 2 or z_out(root, col) == 0)
                if z_out(root, col) == 2 and z_out(root, row) == 2:
                    apply("CNOT", (col, row))

            if z_out(root, root) == 3:
                apply("S", (root,))

            if z_out(root, root) != 2:
                apply("H", (root,))

            if x_out(root, root) != 1:
                apply("S", (root,))

            assert z_out(root, root) == 2 and x_out(root, root) == 1, \
                f"Expected root (2, 1), but got: ({z_out(root, root)}, {x_out(root, root)})"
            # We have no interaction on the tableau anymore
            return root

        def x_out(row, column):
            return remaining.tableau[row, column] + \
                   2 * remaining.tableau[row, column + self.n_qubits]

        def z_out(row, column):
            return remaining.tableau[row + self.n_qubits, column] + \
                   2 * remaining.tableau[row + self.n_qubits, column + self.n_qubits]

        def pick_pivot_row(pivot_col, remaining_columns):
            pivot_rows = []
            for p_r in remaining_columns:
                px = x_out(pivot_col, p_r)
                pz = z_out(pivot_col, p_r)
                if px != 0 and pz != 0 and px != pz:
                    pivot_rows.append((p_r, topo.dist(pivot_col, p_r)))
            assert len(pivot_rows) > 0
            # semi cryptic statement to just get the row index not the distance acc to the architecture
            return min(pivot_rows, key=lambda x: x[1])[0]

        reduction_order = produce_reduction_order(topo)
        for idx, pivot in enumerate(reduction_order):
            remaining_qubits = reduction_order[idx:]
            pivot_row = pick_pivot_row(pivot, remaining_qubits)

            steiner_reduce_column(pivot, pivot_row, remaining_qubits)
            # removed_nodes.append(pivot) # TODO remove later

        signs_copy_z = remaining.signs[self.n_qubits:2 * self.n_qubits].copy()
        for col in range(self.n_qubits):
            if signs_copy_z[col] != 0:
                apply("H", (col,))

        for col in range(self.n_qubits):
            if signs_copy_z[col] != 0:
                apply("S", (col,))
                apply("S", (col,))

        for col in range(self.n_qubits):
            if signs_copy_z[col] != 0:
                apply("H", (col,))

        for col in range(self.n_qubits):
            if remaining.signs[col] != 0:
                apply("S", (col,))
                apply("S", (col,))

        return qc

    def to_clifford_circuit(self):
        qc = QuantumCircuit(self.n_qubits)

        remaining = self.inverse()

        def apply(gate_name: str, gate_data: tuple):
            if gate_name == "CNOT":
                remaining.apply_cnot(gate_data[0], gate_data[1])
                qc.cx(gate_data[0], gate_data[1])
            elif gate_name == "H":
                remaining.apply_h(gate_data[0])
                qc.h(gate_data[0])
            elif gate_name == "S":
                remaining.apply_s(gate_data[0])
                qc.s(gate_data[0])
            else:
                raise Exception("Unknown Gate")

        def x_out(inp, out):
            return remaining.tableau[inp, out] + \
                   2 * remaining.tableau[inp, out + self.n_qubits]

        def z_out(inp, out):
            return remaining.tableau[inp + self.n_qubits, out] + \
                   2 * remaining.tableau[inp + self.n_qubits, out + self.n_qubits]

        n = self.n_qubits
        for col in range(n):
            pivot_row = None
            for row in range(col, n):
                px = x_out(col, row)
                pz = z_out(col, row)
                if px != 0 and pz != 0 and px != pz:
                    pivot_row = row
                    break
            assert pivot_row is not None
            # Move the pivot to the diagonal
            if pivot_row != col:
                apply("CNOT", (pivot_row, col))
                apply("CNOT", (col, pivot_row))
                apply("CNOT", (pivot_row, col))

            # Transform the pivot to the XZ plane
            if z_out(col, col) == 3:
                apply("S", (col,))

            if z_out(col, col) != 2:
                apply("H", (col,))

            if x_out(col, col) != 1:
                apply("S", (col,))

            # Use the pivot to remove all other terms in the X observable.
            for row in range(col + 1, n):
                if x_out(col, row) == 3:
                    apply("S", (row,))

            for row in range(col + 1, n):
                if x_out(col, row) == 2:
                    apply("H", (row,))

            for row in range(col + 1, n):
                if x_out(col, row) != 0:
                    apply("CNOT", (col, row))

            # Use the pivot to remove all other terms in the Z observable.
            for row in range(col + 1, n):
                if z_out(col, row) == 3:
                    apply("S", (row,))

            for row in range(col + 1, n):
                if z_out(col, row) == 1:
                    apply("H", (row,))

            for row in range(col + 1, n):
                if z_out(col, row) != 0:
                    apply("CNOT", (row, col))

        signs_copy_z = remaining.signs[n:2 * n].copy()
        for col in range(n):
            if signs_copy_z[col] != 0:
                apply("H", (col,))

        for col in range(n):
            if signs_copy_z[col] != 0:
                apply("S", (col,))
                apply("S", (col,))

        for col in range(n):
            if signs_copy_z[col] != 0:
                apply("H", (col,))
        for col in range(n):
            if remaining.signs[col] != 0:
                apply("S", (col,))
                apply("S", (col,))

        return qc
