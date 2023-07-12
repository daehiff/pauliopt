import networkx as nx
from qiskit import QuantumCircuit

from pauliopt.pauli.clifford_gates import CX
from pauliopt.pauli.clifford_gates import H, V
from pauliopt.pauli.clifford_tableau import is_cutting, relabel_graph_inplace
from pauliopt.pauli.pauli_circuit import PauliCircuit
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.utils import I, X, Y, Z
from pauliopt.topologies import Topology


def pick_row(pp: PauliPolynomial, columns_to_use, qubits_to_use):
    qubit_scores = []
    for q in qubits_to_use:
        i_score = len([col for col in columns_to_use if pp.pauli_gadgets[col][q] == I])
        x_score = len([col for col in columns_to_use if pp.pauli_gadgets[col][q] == X])
        y_score = len([col for col in columns_to_use if pp.pauli_gadgets[col][q] == Y])
        z_score = len([col for col in columns_to_use if pp.pauli_gadgets[col][q] == Z])
        qubit_scores.append((q, max([i_score, x_score, y_score, z_score])))
    return max(qubit_scores, key=lambda x: x[1])[0]


def pick_next_row(pp: PauliPolynomial, columns_to_use, qubits_to_use):
    qubit_scores = []
    for q in qubits_to_use:
        i_score = len([col for col in columns_to_use if pp.pauli_gadgets[col][q] != I])
        qubit_scores.append((q, i_score))
    return max(qubit_scores, key=lambda x: x[1])[0]


def partition_pauli_polynomial(pp: PauliPolynomial, row: int, columns_to_use: list):
    col_i = []
    col_x = []
    col_y = []
    col_z = []
    for col in columns_to_use:
        if pp.pauli_gadgets[col][row] == I:
            col_i.append(col)
        elif pp.pauli_gadgets[col][row] == X:
            col_x.append(col)
        elif pp.pauli_gadgets[col][row] == Y:
            col_y.append(col)
        elif pp.pauli_gadgets[col][row] == Z:
            col_z.append(col)
        else:
            raise ValueError("Invalid Pauli")
    return col_i, col_x, col_y, col_z


def partition_pauli_polynomial_(pp: PauliPolynomial, row: int, columns_to_use: list):
    col_i, col_x, col_y, col_z = partition_pauli_polynomial(pp, row, columns_to_use)

    cols = [
        (X, Y, col_x + col_y, Z, col_z, len(col_x) + len(col_y)),
        (X, Z, col_x + col_z, Y, col_y, len(col_x) + len(col_z)),
        (Y, Z, col_y + col_z, X, col_x, len(col_y) + len(col_z))
    ]
    type_two_1, type_two_2, cols_2, type_col1, col1, _ = max(cols, key=lambda x: x[-1])
    return col_i, type_two_1, type_two_2, cols_2, type_col1, col1


def pauli_polynomial_steiner_gray_nc(pp: PauliPolynomial, topo: Topology):
    remaining_columns = list(range(pp.num_gadgets))
    perm_gadgets = []
    permutation = {k: k for k in range(pp.num_qubits)}
    G = topo.to_nx

    def identity_recurse(columns_to_use, qubits_to_use):
        if not columns_to_use:
            return PauliCircuit(pp.num_qubits)
        row = pick_row(pp, columns_to_use, qubits_to_use)

        pp_i, pp_x, pp_y, pp_z = partition_pauli_polynomial(pp, row, columns_to_use)

        remaining_qubits = [q for q in qubits_to_use if q != row]
        qc_i = identity_recurse(pp_i, remaining_qubits)
        qc_x = p_recurse(pp_x, remaining_qubits, row, X)
        qc_y = p_recurse(pp_y, remaining_qubits, row, Y)
        qc_z = p_recurse(pp_z, remaining_qubits, row, Z)

        qc_out = qc_i + qc_x + qc_y + qc_z
        return qc_out

    def p_recurse(columns_to_use, qubits_to_use, row, rec_type):
        assert rec_type in [X, Y, Z]
        if not columns_to_use:
            return PauliCircuit(pp.num_qubits)
        elif not qubits_to_use:
            return apply_rotation(columns_to_use, row, rec_type)
        G_: nx.Graph = G.subgraph(qubits_to_use + [row])
        neighbours = [q for q in G_.neighbors(row)]

        if not neighbours:
            return PauliCircuit(pp.num_qubits)

        row_next = pick_row(pp, columns_to_use, neighbours)

        pp_i, t_pp2_1, tpp2_2, pp2, t_p1, pp1 = \
            partition_pauli_polynomial_(pp, row_next, columns_to_use)
        remaining_qubits = [q for q in qubits_to_use if q != row_next]

        if not is_cutting(row_next, G_):
            qc_i = identity_recurse(pp_i, remaining_qubits + [row])
        else:
            qc_i = check_identity(pp_i, remaining_qubits, row, row_next, rec_type)

        qc_two = simplify_twp_p(pp2, remaining_qubits, row, rec_type, row_next,
                                t_pp2_1, tpp2_2)
        qc_one = simplify_one_p(pp1, remaining_qubits, row, rec_type, row_next, t_p1)

        qc_out = qc_i + qc_two + qc_one
        return qc_out

    def apply_rotation(columns_to_use, row, rec_type):
        qc = PauliCircuit(pp.num_qubits)
        if rec_type == X:
            qc.h(row)
        elif rec_type == Y:
            qc.v(row)

        for col in columns_to_use:
            qc.rz(pp.pauli_gadgets[col].angle, row)
            perm_gadgets.append(col)

        if rec_type == X:
            qc.h(row)
        elif rec_type == Y:
            qc.vdg(row)
        return qc

    def check_identity(columns_to_use, remaining_qubits, row, row_next, rec_type):
        qc = PauliCircuit(pp.num_qubits)
        if len(columns_to_use) == 0:
            return qc
        if rec_type == X:
            qc.h(row)
            pp.propagate(H(row), columns_to_use)
        elif rec_type == Y:
            qc.vdg(row)
            pp.propagate(V(row), columns_to_use)
        qc.cx(row_next, row)
        qc.cx(row, row_next)

        pp.propagate(CX(row_next, row), columns_to_use)
        pp.propagate(CX(row, row_next), columns_to_use)
        qc_i = p_recurse(columns_to_use, remaining_qubits, row_next, Z)
        qc += qc_i

        qc.cx(row, row_next)
        qc.cx(row_next, row)
        if rec_type == X:
            qc.h(row)
        elif rec_type == Y:
            qc.v(row)
        return qc

    def simplify_twp_p(columns_to_use, remaining_qubits, row, rec_type, row_next,
                       rec_type_next_1, rec_type_next_2):
        qc = PauliCircuit(pp.num_qubits)
        if len(columns_to_use) == 0:
            return qc

        if rec_type == X:
            qc.h(row)
            pp.propagate(H(row), columns_to_use)
        elif rec_type == Y:
            qc.vdg(row)
            pp.propagate(V(row), columns_to_use)

        if rec_type_next_1 == X and rec_type_next_2 == Y:
            qc.h(row_next)
            pp.propagate(H(row_next), columns_to_use)
        elif rec_type_next_1 == X and rec_type_next_2 == Z:
            qc.vdg(row_next)
            pp.propagate(V(row_next), columns_to_use)
        elif rec_type_next_1 == Y and rec_type_next_2 == Z:
            pass  # just to note this case

        qc.cx(row, row_next)
        pp.propagate(CX(row, row_next), columns_to_use)

        columns_to_use_1 = [col for col in columns_to_use if pp[col][row_next] == Z]
        columns_to_use_2 = [col for col in columns_to_use if pp[col][row_next] == Y]
        qc_z = p_recurse(columns_to_use_1, remaining_qubits, row_next, Z)
        qc_y = p_recurse(columns_to_use_2, remaining_qubits, row_next, Y)
        qc += qc_z
        qc += qc_y
        qc.cx(row, row_next)

        if rec_type == X:
            qc.h(row)
        elif rec_type == Y:
            qc.v(row)

        if rec_type_next_1 == X and rec_type_next_2 == Y:
            qc.h(row_next)
        elif rec_type_next_1 == X and rec_type_next_2 == Z:
            qc.v(row_next)
        return qc

    def simplify_one_p(columns_to_use, remaining_qubits, row, rec_type, row_next,
                       rec_type_next):
        qc = PauliCircuit(pp.num_qubits)
        if len(columns_to_use) == 0:
            return qc

        if rec_type == X:
            qc.h(row)
            pp.propagate(H(row), columns_to_use)
        elif rec_type == Y:
            qc.vdg(row)
            pp.propagate(V(row), columns_to_use)

        if rec_type_next == X:
            qc.h(row_next)
            pp.propagate(H(row_next), columns_to_use)
        elif rec_type_next == Y:
            qc.vdg(row_next)
            pp.propagate(V(row_next), columns_to_use)

        qc.cx(row, row_next)
        pp.propagate(CX(row, row_next), columns_to_use)

        qc_ = p_recurse(columns_to_use, remaining_qubits, row_next, Z)
        qc += qc_
        qc.cx(row, row_next)

        if rec_type == X:
            qc.h(row)
        elif rec_type == Y:
            qc.v(row)

        if rec_type_next == X:
            qc.h(row_next)
        elif rec_type_next == Y:
            qc.v(row_next)
        return qc

    circ_out = identity_recurse(remaining_columns, list(range(pp.num_qubits)))

    permutation = [permutation[i] for i in range(pp.num_qubits)]
    return circ_out, perm_gadgets, permutation
