import itertools
import math
from functools import partial
from multiprocessing import Pool

import networkx as nx
import numpy as np
import stim
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford


from pauliopt.pauli.pauli_circuit import (
    PauliCircuit,
    CliffordGate,
    CliffordType,
    SingleQubitGate,
    TwoQubitClifford,
    SingleQubitClifford,
)
from pauliopt.topologies import Topology


def mult_paulis(p1, p2, sign1, sign2, n_qubits):
    x_1 = p1[:n_qubits].copy()
    z_1 = p1[n_qubits:].copy()
    x_2 = p2[:n_qubits].copy()
    z_2 = p2[n_qubits:].copy()

    x_1_z_2 = z_1 * x_2
    z_1_x_2 = x_1 * z_2

    ac = (x_1_z_2 + z_1_x_2) % 2

    x_1 = (x_1 + x_2) % 2
    z_1 = (z_1 + z_2) % 2

    x_1_z_2 = ((x_1_z_2 + x_1 + z_1) % 2) * ac
    sign_change = int(((np.sum(ac) + 2 * np.sum(x_1_z_2)) % 4) > 1)
    new_sign = (sign1 + sign2 + sign_change) % 4
    new_p1 = np.concatenate([x_1, z_1])
    return new_p1, new_sign


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


def traversal_sum(pivot, traversal, remaining: "CliffordTableau"):
    score = 0
    for child, parent in traversal:
        x_child = remaining.x_out(pivot, child) != 0
        x_parent = remaining.x_out(pivot, parent) != 0

        z_child = remaining.z_out(pivot, child) != 0
        z_parent = remaining.z_out(pivot, parent) != 0
        score += (x_child == x_parent) + (z_child == z_parent)
    return score


def pick_col(
    G,
    remaining: "CliffordTableau",
    possible_swaps,
    include_swaps,
    pivot_row,
    choice_fn=min,
):
    scores = []
    has_cutting_swappable = any([not is_cutting(i, G) for i in possible_swaps])
    for col in G.nodes:
        if not is_cutting(col, G) or (
            include_swaps and has_cutting_swappable and col in possible_swaps
        ):
            row_x = [
                nx.shortest_path_length(G, source=col, target=other_col)
                for other_col in G.nodes
                if remaining.x_out(pivot_row, other_col) != 0
            ]
            row_z = [
                nx.shortest_path_length(G, source=col, target=other_col)
                for other_col in G.nodes
                if remaining.z_out(pivot_row, other_col) != 0
            ]
            dist_x = sum(row_x)
            dist_z = sum(row_z)
            scores.append((col, dist_x + dist_z))

    return choice_fn(scores, key=lambda x: x[1])[0]


def pick_pivot(
    G,
    remaining: "CliffordTableau",
    possible_swaps,
    include_swaps,
    choice_fn=min,
):
    scores = []
    has_cutting_swappable = any([not is_cutting(i, G) for i in possible_swaps])
    for col in G.nodes:
        if not is_cutting(col, G) or (
            include_swaps and has_cutting_swappable and col in possible_swaps
        ):
            row_x = [
                nx.shortest_path_length(G, source=col, target=other_col)
                for other_col in G.nodes
                if remaining.x_out(col, other_col) != 0
            ]
            row_z = [
                nx.shortest_path_length(G, source=col, target=other_col)
                for other_col in G.nodes
                if remaining.z_out(col, other_col) != 0
            ]
            dist_x = sum(row_x)
            dist_z = sum(row_z)
            scores.append((col, dist_x + dist_z))

    return choice_fn(scores, key=lambda x: x[1])[0]


def pick_row(G, remaining: "CliffordTableau", remaining_rows, choice_fn=min):
    scores = []
    for row in remaining_rows:
        row_x = [1 for col in G.nodes if remaining.x_out(row, col) != 0]
        row_z = [1 for col in G.nodes if remaining.z_out(row, col) != 0]
        dist_x = sum(row_x)
        dist_z = sum(row_z)
        scores.append((row, dist_x + dist_z))

    return choice_fn(scores, key=lambda x: x[1])[0]


def update_dfs(dfs, parent, child):
    for i, (p, c) in enumerate(dfs):
        if c == parent:
            dfs[i] = (p, child)
        if p == parent:
            dfs[i] = (child, c)
        if c == child:
            dfs[i] = (p, parent)
        if p == child:
            dfs[i] = (parent, c)
    return dfs


def relabel_graph_inplace(G, parent, child):
    swap = {parent: -1}
    nx.relabel_nodes(G, swap, copy=False)
    swap = {child: parent}
    nx.relabel_nodes(G, swap, copy=False)
    swap = {-1: child}
    nx.relabel_nodes(G, swap, copy=False)


def compute_steiner_tree(
    root: int,
    nodes: [int],
    sub_graph: nx.Graph,
    lookup: dict,
    swappable_nodes,
    permutation,
    remaining: "CliffordTableau",
    include_swaps,
):
    steiner_stree = nx.algorithms.approximation.steinertree.steiner_tree(
        sub_graph, nodes
    )
    steiner_stree = nx.Graph(steiner_stree)
    if len(steiner_stree.nodes()) < 1:
        return []
    if include_swaps:
        for _ in range(remaining.n_qubits):
            dfs = list(reversed(list(nx.dfs_edges(steiner_stree, source=root))))
            swapped = []
            while dfs:
                parent, child = dfs.pop(0)
                if parent == root:
                    continue
                if (
                    lookup[parent] == 0
                    and lookup[child] == 1
                    and child in swappable_nodes
                    and parent in swappable_nodes
                ):
                    relabel_graph_inplace(steiner_stree, parent, child)
                    relabel_graph_inplace(sub_graph, parent, child)
                    dfs = update_dfs(dfs, parent, child)
                    # remaining.swap_cols(parent, child)
                    permutation[parent], permutation[child] = (
                        permutation[child],
                        permutation[parent],
                    )

                    swapped.append(parent)
                    swapped.append(child)

        steiner_stree = nx.algorithms.approximation.steinertree.steiner_tree(
            sub_graph, nodes
        )
    traversal = nx.bfs_edges(steiner_stree, source=root)
    return list(reversed(list(traversal)))


def sanitize_field_z(row, column, remaining, apply):
    if remaining.z_out(row, column) == 3:
        apply("S", (column,))

    if remaining.z_out(row, column) == 1:
        apply("H", (column,))


def sanitize_field_x(row, column, remaining, apply):
    if remaining.x_out(row, column) == 3:
        apply("S", (column,))

    if remaining.x_out(row, column) == 2:
        apply("H", (column,))


def steiner_up_down_process_z(
    pivot_col,
    pivot_row,
    row_z,
    sub_graph,
    remaining,
    apply,
    swappable_nodes,
    permutation,
    include_swaps,
):
    row_z_ = list(set([pivot_col] + row_z))
    lookup = {
        node: int(remaining.z_out(pivot_row, node) != 0) for node in sub_graph.nodes
    }
    traversal = compute_steiner_tree(
        pivot_col,
        row_z_,
        sub_graph,
        lookup,
        swappable_nodes,
        permutation,
        remaining,
        include_swaps,
    )
    for parent, child in traversal:
        if remaining.z_out(pivot_row, parent) == 0:
            apply("CNOT", (parent, child))

    for parent, child in traversal:
        apply("CNOT", (child, parent))


def steiner_up_down_process_x(
    pivot_col,
    pivot_row,
    row_x,
    sub_graph,
    remaining,
    apply,
    swappable_nodes,
    permutation,
    include_swaps,
):
    row_x_ = list(set([pivot_col] + row_x))
    lookup = {
        node: int(remaining.x_out(pivot_row, node) != 0) for node in sub_graph.nodes
    }
    traversal = compute_steiner_tree(
        pivot_col,
        row_x_,
        sub_graph,
        lookup,
        swappable_nodes,
        permutation,
        remaining,
        include_swaps,
    )
    for parent, child in traversal:
        if remaining.x_out(pivot_row, parent) == 0:
            apply("CNOT", (child, parent))

    for parent, child in traversal:
        apply("CNOT", (parent, child))


def steiner_reduce_column(
    pivot, sub_graph, remaining, apply, swappable_nodes, permutation, include_swaps
):
    row_x = [col for col in sub_graph.nodes if remaining.x_out(
        pivot, col) != 0]
    for col in row_x:
        sanitize_field_x(pivot, col, remaining, apply)

    steiner_up_down_process_x(
        pivot,
        pivot,
        row_x,
        sub_graph,
        remaining,
        apply,
        swappable_nodes,
        permutation,
        include_swaps,
    )

    row_z = [row for row in sub_graph.nodes if remaining.z_out(
        pivot, row) != 0]
    for col in row_z:
        sanitize_field_z(pivot, col, remaining, apply)
    if remaining.x_out(pivot, pivot) == 3:
        apply("S", (pivot,))
    steiner_up_down_process_z(
        pivot,
        pivot,
        row_z,
        sub_graph,
        remaining,
        apply,
        swappable_nodes,
        permutation,
        include_swaps,
    )

    # ensure that the pivots are in ZX basis

    assert remaining.x_out(pivot, pivot) == 1
    assert remaining.z_out(pivot, pivot) == 2
    # if remaining.z_out(pivot, pivot) == 3:
    #     apply("S", (pivot,))
    #
    # if remaining.z_out(pivot, pivot) != 2:
    #     apply("H", (pivot,))
    #
    # if remaining.x_out(pivot, pivot) != 1:
    #     apply("S", (pivot,))


def steiner_reduce_column_perm_row_col(
    pivot_col,
    pivot_row,
    sub_graph,
    remaining,
    apply,
    swappable_nodes,
    permutation,
    include_swaps,
):
    row_x = [col for col in sub_graph.nodes if remaining.x_out(
        pivot_row, col) != 0]
    for col in row_x:
        sanitize_field_x(pivot_row, col, remaining, apply)

    # TODO check this one
    steiner_up_down_process_x(
        pivot_col,
        pivot_row,
        row_x,
        sub_graph,
        remaining,
        apply,
        swappable_nodes,
        permutation,
        include_swaps,
    )

    row_z = [row for row in sub_graph.nodes if remaining.z_out(
        pivot_row, row) != 0]
    for col in row_z:
        sanitize_field_z(pivot_row, col, remaining, apply)

    if remaining.x_out(pivot_row, pivot_col) == 3:
        apply("S", (pivot_col,))

    steiner_up_down_process_z(
        pivot_col,
        pivot_row,
        row_z,
        sub_graph,
        remaining,
        apply,
        swappable_nodes,
        permutation,
        include_swaps,
    )

    # ensure that the pivots are in ZX basis

    # assert remaining.x_out(pivot_col, pivot_col) == 1
    # assert remaining.z_out(pivot_col, pivot_col) == 2
    # if remaining.z_out(pivot, pivot) == 3:
    #     apply("S", (pivot,))
    #
    # if remaining.z_out(pivot, pivot) != 2:
    #     apply("H", (pivot,))
    #
    # if remaining.x_out(pivot, pivot) != 1:
    #     apply("S", (pivot,))


def optimal_remove_signs(
    qc,
    remaining,
    apply,
):
    signs_copy_z = remaining.signs[remaining.n_qubits: 2 *
                                   remaining.n_qubits].copy()
    for col in range(remaining.n_qubits):
        if signs_copy_z[col] != 0:
            apply(qc, remaining, "H", (col,))

    for col in range(remaining.n_qubits):
        if signs_copy_z[col] != 0:
            apply(qc, remaining, "S", (col,))
            apply(qc, remaining, "S", (col,))

    for col in range(remaining.n_qubits):
        if signs_copy_z[col] != 0:
            apply(qc, remaining, "H", (col,))

    for col in range(remaining.n_qubits):
        if remaining.signs[col] != 0:
            apply(qc, remaining, "S", (col,))
            apply(qc, remaining, "S", (col,))


def optimal_pivot_removal(qc, remaining, apply, pivot):
    pass


class CliffordTableau:
    def __init__(
        self, n_qubits: int = None, tableau: np.array = None, signs: np.array = None
    ):
        if n_qubits is None and tableau is None:
            raise Exception(
                "Either Tableau or number of qubits must be defined")
        if tableau is None:
            self.tableau = np.eye(2 * n_qubits)
            self.signs = np.zeros((2 * n_qubits))
            self.n_qubits = n_qubits
        else:
            if tableau.shape[0] != tableau.shape[1]:
                raise Exception(
                    f"Must be a 2nx2n Tableau, but is of shape: {tableau.shape}"
                )
            self.n_qubits = int(tableau.shape[1] / 2.0)
            self.tableau = tableau
            if signs is None:
                self.signs = np.zeros((2 * self.n_qubits))
            else:
                self.signs = signs
            if not 2 * self.n_qubits == tableau.shape[1]:
                raise Exception(
                    f"Tableau of shape: {tableau.shape}, is not a factor of 2"
                )

    def copy(self):
        return CliffordTableau(tableau=self.tableau.copy(), signs=self.signs.copy())

    def prepend_h(self, qubit):
        self.signs[[qubit, self.n_qubits + qubit]] = self.signs[
            [self.n_qubits + qubit, qubit]
        ]
        self.tableau[[self.n_qubits + qubit, qubit], :] = self.tableau[
            [qubit, self.n_qubits + qubit], :
        ]

    def append_h(self, qubit):
        self.signs = (
            self.signs + self.tableau[:, qubit] *
            self.tableau[:, self.n_qubits + qubit]
        ) % 2

        self.tableau[:, [self.n_qubits + qubit, qubit]] = self.tableau[
            :, [qubit, self.n_qubits + qubit]
        ]

    def prepend_s(self, qubit):
        stabilizer = self.tableau[qubit, :]
        destabilizer = self.tableau[qubit + self.n_qubits, :]
        stab_sign = self.signs[qubit]
        destab_sign = self.signs[qubit + self.n_qubits]

        destabilizer, destab_sign = mult_paulis(
            stabilizer, destabilizer, stab_sign, destab_sign, self.n_qubits
        )
        self.insert_pauli_row(destabilizer, destab_sign, qubit)

    def append_s(self, qubit):
        self.signs = (
            self.signs + self.tableau[:, qubit] *
            self.tableau[:, self.n_qubits + qubit]
        ) % 2

        self.tableau[:, self.n_qubits + qubit] = (
            self.tableau[:, self.n_qubits + qubit] + self.tableau[:, qubit]
        ) % 2

    def prepend_cnot(self, control, target):
        stab_ctrl = self.tableau[control, :]
        stab_target = self.tableau[target, :]
        stab_sign_ctrl = self.signs[control]
        stab_sign_target = self.signs[target]

        destab_ctrl = self.tableau[control + self.n_qubits, :]
        destab_target = self.tableau[target + self.n_qubits, :]
        destab_sign_ctrl = self.signs[control + self.n_qubits]
        destab_sign_target = self.signs[target + self.n_qubits]

        stab_ctrl, stab_sign_ctrl = mult_paulis(
            stab_ctrl, stab_target, stab_sign_ctrl, stab_sign_target, self.n_qubits
        )

        destab_target, destab_sign_target = mult_paulis(
            destab_target,
            destab_ctrl,
            destab_sign_target,
            destab_sign_ctrl,
            self.n_qubits,
        )

        self.insert_pauli_row(stab_ctrl, stab_sign_ctrl, control)
        self.insert_pauli_row(
            destab_target, destab_sign_target, target + self.n_qubits)

    def append_cnot(self, control, target):
        x_ia = self.tableau[:, control]
        x_ib = self.tableau[:, target]

        z_ia = self.tableau[:, self.n_qubits + control]
        z_ib = self.tableau[:, self.n_qubits + target]

        self.tableau[:, target] = (
            self.tableau[:, target] + self.tableau[:, control]
        ) % 2
        self.tableau[:, self.n_qubits + control] = (
            self.tableau[:, self.n_qubits + control]
            + self.tableau[:, self.n_qubits + target]
        ) % 2

        tmp_sum = ((x_ib + z_ia) % 2 + np.ones(z_ia.shape)) % 2
        self.signs = (self.signs + x_ia * z_ib * tmp_sum) % 2

    def append_v(self, qubit):
        self.append_h(qubit)
        self.append_s(qubit)
        self.append_h(qubit)

    def prepend_v(self, qubit):
        self.prepend_h(qubit)
        self.prepend_s(qubit)
        self.prepend_h(qubit)

    def swap_rows(self, a, b):
        # swap in stabilizer basis
        self.tableau[[a, b], :] = self.tableau[[b, a], :]
        self.signs[[a, b]] = self.signs[[b, a]]

        # swap in destabilizer basis
        self.tableau[[a + self.n_qubits, b + self.n_qubits], :] = self.tableau[
            [b + self.n_qubits, a + self.n_qubits], :
        ]
        self.signs[[a + self.n_qubits, b + self.n_qubits]] = self.signs[
            [b + self.n_qubits, a + self.n_qubits]
        ]

    def insert_pauli_row(self, pauli, p_sing, row):
        for i in range(self.n_qubits):
            if (self.tableau[row, i] + pauli[i]) % 2 == 1:
                self.tableau[row, i] = (self.tableau[row, i] + 1) % 2

            if (
                self.tableau[row, i + self.n_qubits] + pauli[i + self.n_qubits]
            ) % 2 == 1:
                self.tableau[row, i + self.n_qubits] = (
                    self.tableau[row, i + self.n_qubits] + 1
                ) % 2
        if (self.signs[row] + p_sing) % 2 == 1:
            self.signs[row] = (self.signs[row] + 1) % 2

    def x_out(self, row, col):
        return self.tableau[row, col] + 2 * self.tableau[row, col + self.n_qubits]

    def z_out(self, row, col):
        return (
            self.tableau[row + self.n_qubits, col]
            + 2 * self.tableau[row + self.n_qubits, col + self.n_qubits]
        )

    @property
    def x_matrix(self):
        x_matrx = np.zeros((self.n_qubits, self.n_qubits), dtype=int)
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                x_matrx[i, j] = (
                    self.tableau[i, j] + 2 * self.tableau[i, j + self.n_qubits]
                )
        return x_matrx

    @property
    def z_matrix(self):
        z_matrx = np.zeros((self.n_qubits, self.n_qubits), dtype=int)
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                z_matrx[i, j] = (
                    1 * self.tableau[i + self.n_qubits, j]
                    + 2 * self.tableau[i + self.n_qubits, j + self.n_qubits]
                )
        return z_matrx

    @property
    def A(self):
        A = self.x_matrix + self.z_matrix
        return A

    @property
    def x(self):
        return self.tableau[0: self.n_qubits, :].astype(int)

    @property
    def z(self):
        return self.tableau[:, self.n_qubits: 2 * self.n_qubits].astype(bool)

    @staticmethod
    def get_lookup():
        lookup = np.zeros((2, 2, 2, 2), dtype=int)
        lookup[0, 1, 1, 0] = lookup[1, 0, 1, 1] = lookup[1, 1, 0, 1] = -1
        lookup[0, 1, 1, 1] = lookup[1, 0, 0, 1] = lookup[1, 1, 1, 0] = 1
        lookup.setflags(write=False)
        return lookup

    def prepend_gate(self, gate: CliffordGate):
        if gate.clifford_type == CliffordType.H:
            assert isinstance(gate, SingleQubitClifford)
            self.prepend_h(gate.qubit)
        elif gate.clifford_type == CliffordType.S:
            assert isinstance(gate, SingleQubitClifford)
            self.prepend_s(gate.qubit)
        elif gate.clifford_type == CliffordType.V:
            assert isinstance(gate, SingleQubitClifford)
            self.prepend_h(gate.qubit)
            self.prepend_s(gate.qubit)
            self.prepend_h(gate.qubit)
        elif gate.clifford_type == CliffordType.CX:
            assert isinstance(gate, TwoQubitClifford)
            self.prepend_cnot(gate.control, gate.target)
        elif gate.clifford_type == CliffordType.CY:
            assert isinstance(gate, TwoQubitClifford)
            self.prepend_s(gate.target)
            self.prepend_cnot(gate.control, gate.target)
            self.prepend_s(gate.target)
            self.prepend_s(gate.target)
            self.prepend_s(gate.target)
        elif gate.clifford_type == CliffordType.CZ:
            assert isinstance(gate, TwoQubitClifford)
            self.prepend_h(gate.target)
            self.prepend_cnot(gate.control, gate.target)
            self.prepend_h(gate.target)
        else:
            raise ValueError("Invalid Clifford gate type")

    def append_gate(self, gate: CliffordGate):
        if gate.clifford_type == CliffordType.H:
            assert isinstance(gate, SingleQubitClifford)
            self.append_h(gate.qubit)
        elif gate.clifford_type == CliffordType.S:
            assert isinstance(gate, SingleQubitClifford)
            self.append_s(gate.qubit)
        elif gate.clifford_type == CliffordType.Sdg:
            assert isinstance(gate, SingleQubitClifford)
            self.append_s(gate.qubit)
            self.append_s(gate.qubit)
            self.append_s(gate.qubit)
        elif gate.clifford_type == CliffordType.V:
            assert isinstance(gate, SingleQubitClifford)
            self.append_h(gate.qubit)
            self.append_s(gate.qubit)
            self.append_h(gate.qubit)
        elif gate.clifford_type == CliffordType.Vdg:
            assert isinstance(gate, SingleQubitClifford)
            self.append_h(gate.qubit)
            self.append_s(gate.qubit)
            self.append_s(gate.qubit)
            self.append_s(gate.qubit)
            self.append_h(gate.qubit)
        elif gate.clifford_type == CliffordType.CX:
            assert isinstance(gate, TwoQubitClifford)
            self.append_cnot(gate.control, gate.target)
        elif gate.clifford_type == CliffordType.CY:
            assert isinstance(gate, TwoQubitClifford)
            self.append_s(gate.target)
            self.append_s(gate.target)
            self.append_s(gate.target)
            self.append_cnot(gate.control, gate.target)
            self.append_s(gate.target)
        elif gate.clifford_type == CliffordType.CZ:
            assert isinstance(gate, TwoQubitClifford)
            self.append_h(gate.target)
            self.append_cnot(gate.control, gate.target)
            self.append_h(gate.target)

        else:
            raise TypeError(
                f"Unrecongnized Gate type: {type(gate)} for Clifford Tableaus"
            )

    def append_circuit(self, qc: QuantumCircuit):
        for op in qc:
            if op.operation.name == "h":
                self.append_h(op.qubits[0].index)
            elif op.operation.name == "s":
                self.append_s(op.qubits[0].index)
            elif op.operation.name == "cx":
                self.append_cnot(op.qubits[0].index, op.qubits[1].index)

    def prepend_circuit(self, qc: QuantumCircuit):
        for op in reversed(qc):
            if op.operation.name == "h":
                self.prepend_h(op.qubits[0].index)
            elif op.operation.name == "s":
                self.prepend_s(op.qubits[0].index)
            elif op.operation.name == "cx":
                self.prepend_cnot(op.qubits[0].index, op.qubits[1].index)

    @staticmethod
    def from_qiskit(tableau: Clifford):
        out = CliffordTableau(n_qubits=tableau.num_qubits)
        out.tableau = tableau.symplectic_matrix.astype(np.float64)
        out.signs = tableau.phase.astype(np.float64)
        return out

    @staticmethod
    def from_circuit(circ: QuantumCircuit):
        tableau = CliffordTableau(n_qubits=circ.num_qubits)
        for op in circ:
            if op.operation.name == "h":
                tableau.append_h(op.qubits[0]._index)
            elif op.operation.name == "s":
                tableau.append_s(op.qubits[0]._index)
            elif op.operation.name == "cx":
                tableau.append_cnot(op.qubits[0]._index, op.qubits[1]._index)
            else:
                raise TypeError(
                    f"Unrecongnized Gate type: {op.operation.name} for Clifford Tableaus"
                )
        return tableau

    def inverse(self):
        # TODO: Implement inverse

        x2x = self.tableau[: self.n_qubits,
                           : self.n_qubits].copy().astype(np.bool8)
        z2z = (
            self.tableau[
                self.n_qubits: 2 * self.n_qubits, self.n_qubits: 2 * self.n_qubits
            ]
            .copy()
            .astype(np.bool8)
        )
        x2z = (
            self.tableau[: self.n_qubits, self.n_qubits: 2 * self.n_qubits]
            .copy()
            .astype(np.bool8)
        )
        z2x = (
            self.tableau[self.n_qubits: 2 * self.n_qubits, : self.n_qubits]
            .copy()
            .astype(np.bool8)
        )
        tab = stim.Tableau.from_numpy(
            x2x=x2x,
            x2z=x2z,
            z2x=z2x,
            z2z=z2z,
            x_signs=self.signs[0: self.n_qubits].copy().astype(np.bool8),
            z_signs=self.signs[self.n_qubits: 2 * self.n_qubits]
            .copy()
            .astype(np.bool8),
        )
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

    def to_clifford_circuit_arch_aware(self, topo: Topology, include_swaps: bool = True):
        qc = PauliCircuit(self.n_qubits)

        remaining = self.inverse()
        permutation = {v: v for v in range(self.n_qubits)}
        swappable_nodes = list(range(self.n_qubits))

        G = topo.to_nx
        for e1, e2 in G.edges:
            G[e1][e2]["weight"] = 0

        def apply(gate_name: str, gate_data: tuple):
            if gate_name == "CNOT":
                remaining.append_cnot(gate_data[0], gate_data[1])
                qc.cx(gate_data[0], gate_data[1])
                if gate_data[0] in swappable_nodes:
                    swappable_nodes.remove(gate_data[0])
                if gate_data[1] in swappable_nodes:
                    swappable_nodes.remove(gate_data[1])
                G[gate_data[0]][gate_data[1]]["weight"] = 2
            elif gate_name == "H":
                remaining.append_h(gate_data[0])
                qc.h(gate_data[0])
            elif gate_name == "S":
                remaining.append_s(gate_data[0])
                qc.s(gate_data[0])
            else:
                raise Exception("Unknown Gate")

        while G.nodes:
            pivot_col = pick_pivot(
                G, remaining, swappable_nodes, include_swaps)

            if is_cutting(pivot_col, G) and include_swaps:
                non_cutting_vectices = [
                    (
                        node,
                        nx.shortest_path_length(
                            G, source=node, target=pivot_col, weight="weight"
                        ),
                    )
                    for node in G.nodes
                    if not is_cutting(node, G) and node in swappable_nodes
                ]
                non_cutting = min(non_cutting_vectices, key=lambda x: x[1])[0]

                relabel_graph_inplace(G, non_cutting, pivot_col)
                # remaining.swap_cols(parent, child)
                permutation[pivot_col], permutation[non_cutting] = (
                    permutation[non_cutting],
                    permutation[pivot_col],
                )

            steiner_reduce_column(
                pivot_col,
                G,
                remaining,
                apply,
                swappable_nodes,
                permutation,
                include_swaps,
            )

            if pivot_col in swappable_nodes:
                swappable_nodes.remove(pivot_col)
            G.remove_node(pivot_col)

        signs_copy_z = remaining.signs[self.n_qubits: 2 * self.n_qubits].copy()
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

        permutation = [permutation[i] for i in range(self.n_qubits)]

        return qc, permutation

    def to_clifford_circuit_arch_aware_qiskit(
        self, topo: Topology, include_swaps: bool = True
    ):
        circ, perm = self.to_clifford_circuit_arch_aware(topo, include_swaps)
        return circ.to_qiskit(), perm

    def _optimal_synth(self, topo: Topology, order):
        qc = QuantumCircuit(self.n_qubits)
        remaining = self.inverse()

        def apply(gate_name: str, gate_data: tuple):
            if gate_name == "CNOT":
                remaining.append_cnot(gate_data[0], gate_data[1])
                qc.cx(gate_data[0], gate_data[1])
            elif gate_name == "H":
                remaining.append_h(gate_data[0])
                qc.h(gate_data[0])
            elif gate_name == "S":
                remaining.append_s(gate_data[0])
                qc.s(gate_data[0])
            else:
                raise Exception("Unknown Gate")

        G = topo.to_nx
        for pivot in order:
            if is_cutting(pivot, G):
                return None
            steiner_reduce_column(pivot, G, remaining, apply, [], [], False)
            G.remove_node(pivot)

        signs_copy_z = remaining.signs[self.n_qubits: 2 * self.n_qubits].copy()
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

    def optimal_to_circuit(self, topo: Topology):
        seed = list(range(self.n_qubits))
        final_circuits = []
        for comb in itertools.permutations(seed):
            circ = self._optimal_synth(topo, comb)
            if circ is not None:
                cx_count = circ.count_ops(
                )["cx"] if "cx" in circ.count_ops() else 0
                final_circuits.append((circ, cx_count))
        print(list(sorted([x[1] for x in final_circuits])))
        return min(final_circuits, key=lambda x: x[1])[0]

    def to_clifford_circuit(self):
        qc = QuantumCircuit(self.n_qubits)

        remaining = self.inverse()

        def apply(gate_name: str, gate_data: tuple):
            if gate_name == "CNOT":
                remaining.append_cnot(gate_data[0], gate_data[1])
                qc.cx(gate_data[0], gate_data[1])
            elif gate_name == "H":
                remaining.append_h(gate_data[0])
                qc.h(gate_data[0])
            elif gate_name == "S":
                remaining.append_s(gate_data[0])
                qc.s(gate_data[0])
            else:
                raise Exception("Unknown Gate")

        def x_out(inp, out):
            return (
                remaining.tableau[inp, out]
                + 2 * remaining.tableau[inp, out + self.n_qubits]
            )

        def z_out(inp, out):
            return (
                remaining.tableau[inp + self.n_qubits, out]
                + 2 * remaining.tableau[inp +
                                        self.n_qubits, out + self.n_qubits]
            )

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

        signs_copy_z = remaining.signs[n: 2 * n].copy()
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
