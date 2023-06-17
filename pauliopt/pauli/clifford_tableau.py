import networkx as nx
import numpy as np
import stim
from qiskit import QuantumCircuit

from pauliopt.pauli.clifford_gates import CliffordGate, CliffordType, SingleQubitGate, \
    ControlGate
from pauliopt.topologies import Topology
import galois


def reconstruct_tableau(tableau: stim.Tableau):
    xsx, xsz, zsx, zsz, x_signs, z_signs = tableau.to_numpy()
    x_row = np.concatenate([xsx, xsz], axis=1)
    z_row = np.concatenate([zsx, zsz], axis=1)
    return np.concatenate([x_row, z_row], axis=0).astype(np.int64)


def reconstruct_tableau_signs(tableau: stim.Tableau):
    xsx, xsz, zsx, zsz, x_signs, z_signs = tableau.to_numpy()
    signs = np.concatenate([x_signs, z_signs]).astype(np.int64)
    return signs


def is_cutting(vertex, g):
    return vertex in nx.articulation_points(g)


def get_possible_indices(remaining: "CliffordTableau", sub_graph: nx.Graph):
    for i in sub_graph.nodes:
        for j in sub_graph.nodes:
            yield i, j


def pick_pivots(G, remaining: "CliffordTableau"):
    scores = []
    possible_indices = get_possible_indices(remaining, G)
    for col, row in possible_indices:
        if not is_cutting(col, G):
            x_row = list(
                set([v for v in G.nodes if remaining.x_out(row, v) != 0] + [col]))
            z_row = list(
                set([v for v in G.nodes if remaining.z_out(row, v) != 0] + [col]))

            dist_x = len(compute_steiner_tree(col, x_row, G))
            dist_z = len(compute_steiner_tree(col, z_row, G))
            # dist_x = sum([nx.shortest_path_length(G, col, v) for v in G.nodes
            #               if remaining.x_out(row, v) != 0])
            # dist_z = sum([nx.shortest_path_length(G, col, v) for v in G.nodes
            #               if remaining.z_out(row, v) != 0])
            scores.append((col, row, dist_x + dist_z))
    assert len(scores) > 0
    return min(scores, key=lambda x: x[2])[:2]


def compute_steiner_tree(root: int, nodes: [int], sub_graph: nx.Graph):
    # TODO there must be an other more efficient way (there is!)
    steiner_stree = nx.algorithms.approximation.steinertree.steiner_tree(sub_graph, nodes)
    if len(steiner_stree.nodes()) < 1:
        return []
    traversal = nx.bfs_edges(steiner_stree, source=root)
    return list(reversed(list(traversal)))


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

    def swap_rows(self, a, b):
        # swap in stabilizer basis
        self.tableau[[a, b], :] = self.tableau[[b, a], :]
        self.signs[[a, b]] = self.signs[[b, a]]

        # swap in destabilizer basis
        self.tableau[[a + self.n_qubits, b + self.n_qubits], :] = \
            self.tableau[[b + self.n_qubits, a + self.n_qubits], :]
        self.signs[[a + self.n_qubits, b + self.n_qubits]] = \
            self.signs[[b + self.n_qubits, a + self.n_qubits]]

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

    def _apply_cnot_signs(self, control, target):
        x_ia = self.tableau[:, control]
        x_ib = self.tableau[:, target]

        z_ia = self.tableau[:, self.n_qubits + control]
        z_ib = self.tableau[:, self.n_qubits + target]

        tmp_sum = ((x_ib + z_ia) % 2 + np.ones(z_ia.shape)) % 2
        self.signs = (self.signs + x_ia * z_ib * tmp_sum) % 2

    def x_out(self, row, col):
        return self.tableau[row, col] + \
               2 * self.tableau[row, col + self.n_qubits]

    def z_out(self, row, col):
        return self.tableau[row + self.n_qubits, col] + \
               2 * self.tableau[row + self.n_qubits, col + self.n_qubits]

    def append_gate(self, gate: CliffordGate):
        if gate.c_type == CliffordType.H:
            assert isinstance(gate, SingleQubitGate)
            self.apply_h(gate.qubit)
        elif gate.c_type == CliffordType.S:
            assert isinstance(gate, SingleQubitGate)
            self.apply_s(gate.qubit)
        elif gate.c_type == CliffordType.V:
            assert isinstance(gate, SingleQubitGate)
            self.apply_h(gate.qubit)
            self.apply_s(gate.qubit)
            self.apply_h(gate.qubit)
        elif gate.c_type == CliffordType.CX:
            assert isinstance(gate, ControlGate)
            self.apply_cnot(gate.control, gate.target)
        elif gate.c_type == CliffordType.CY:
            assert isinstance(gate, ControlGate)
            self.apply_s(gate.target)
            self.apply_s(gate.target)
            self.apply_s(gate.target)
            self.apply_cnot(gate.control, gate.target)
            self.apply_s(gate.target)
        elif gate.c_type == CliffordType.CZ:
            assert isinstance(gate, ControlGate)
            self.apply_h(gate.target)
            self.apply_cnot(gate.control, gate.target)
            self.apply_h(gate.target)
        else:
            raise TypeError(
                f"Unrecongnized Gate type: {type(gate)} for Clifford Tableaus")

    def append_circuit(self, qc: QuantumCircuit):
        for op in qc:
            if op.operation.name == "h":
                self.apply_h(op.qubits[0].index)
            elif op.operation.name == "s":
                self.apply_s(op.qubits[0].index)
            elif op.operation.name == "cx":
                self.apply_cnot(op.qubits[0].index, op.qubits[1].index)

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
            else:
                raise TypeError(
                    f"Unrecongnized Gate type: {op.operation.name} for Clifford Tableaus")
        return tableau

    def inverse(self):
        # TODO: Implement inverse

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

    @property
    def z_matrix(self):
        z_matrix = np.zeros((self.n_qubits, self.n_qubits), dtype=np.int64)
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                z_matrix[i, j] = self.z_out(i, j)
        return z_matrix

    @property
    def x_matrix(self):
        x_matrix = np.zeros((self.n_qubits, self.n_qubits), dtype=np.int64)
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                x_matrix[i, j] = self.x_out(i, j)
        return x_matrix

    def print_zx(self, X=True, Z=True):

        if X:
            print("X: ")
            for i in range(self.n_qubits):
                for j in range(self.n_qubits):
                    print(f" {int(self.x_out(i, j))} ", end="")
                print()
        if Z:
            print("Z: ")
            for i in range(self.n_qubits):
                for j in range(self.n_qubits):
                    print(f" {int(self.z_out(i, j))} ", end="")
                print()

    def to_cifford_circuit_arch_aware(self, topo: Topology):
        qc = QuantumCircuit(self.n_qubits)

        remaining = self.inverse()
        permutation = list(range(self.n_qubits))

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
            if remaining.z_out(row, column) == 3:
                apply("S", (column,))

            if remaining.z_out(row, column) == 1:
                apply("H", (column,))

        def sanitize_field_x(row, column):
            if remaining.x_out(row, column) == 3:
                apply("S", (column,))

            if remaining.x_out(row, column) == 2:
                apply("H", (column,))

        def steiner_up_down_process_z(pivot, row_z, sub_graph):
            row_z_ = list(set([pivot] + row_z))
            traversal = compute_steiner_tree(pivot, row_z_, sub_graph)
            for parent, child in traversal:
                if remaining.z_out(pivot, parent) == 0:
                    apply("CNOT", (parent, child))

            for parent, child in traversal:
                apply("CNOT", (child, parent))

        def steiner_up_down_process_x(pivot, row_x, sub_graph):
            row_x_ = list(set([pivot] + row_x))
            traversal = compute_steiner_tree(pivot, row_x_, sub_graph)
            for parent, child in traversal:
                if remaining.x_out(pivot, parent) == 0:
                    apply("CNOT", (child, parent))  # TODO check!

            for parent, child in traversal:
                apply("CNOT", (parent, child))

        def steiner_reduce_column(pivot, sub_graph):
            row_x = [col for col in sub_graph.nodes if remaining.x_out(pivot, col) != 0]
            for col in row_x:
                sanitize_field_x(pivot, col)
            steiner_up_down_process_x(pivot, row_x, sub_graph)

            row_z = [col for col in sub_graph.nodes if remaining.z_out(pivot, col) != 0]
            for col in row_z:
                sanitize_field_z(pivot, col)
            if remaining.x_out(pivot, pivot) == 3:
                apply("S", (pivot,))
            steiner_up_down_process_z(pivot, row_z, sub_graph)

            # ensure that the pivots are in ZX basis
            if remaining.z_out(pivot, pivot) == 3:
                apply("S", (pivot,))

            if remaining.z_out(pivot, pivot) != 2:
                apply("H", (pivot,))

            if remaining.x_out(pivot, pivot) != 1:
                apply("S", (pivot,))

        # reduction_order = produce_reduction_order_old(topo, remaining)
        G = topo.to_nx

        remaining_qubits = list(range(self.n_qubits))
        while G.nodes:
            pivot_col, pivot_row = pick_pivots(G, remaining)
            if pivot_col != pivot_row:
                remaining.swap_rows(pivot_row, pivot_col)
            steiner_reduce_column(pivot_col, G)

            # swap the pivot column and the pivot row in the permutation list
            permutation[pivot_col], permutation[pivot_row] = \
                permutation[pivot_row], permutation[pivot_col]

            remaining_qubits.remove(pivot_col)
            G.remove_node(pivot_col)

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
        print(remaining.signs)
        return qc, permutation

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
