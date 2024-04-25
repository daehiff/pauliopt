import networkx as nx


from pauliopt.pauli.clifford_tableau import is_cutting
from pauliopt.pauli.pauli_circuit import PauliCircuit, CX, H, V, Sdg
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.utils import I, X, Y, Z, Pauli
from pauliopt.topologies import Topology


def pick_row(pp: PauliPolynomial, columns_to_use, qubits_to_use):
    qubit_scores = []
    for q in qubits_to_use:
        i_score = len([col for col in columns_to_use if pp.pauli_gadgets[col][q] == I])
        x_score = len([col for col in columns_to_use if pp.pauli_gadgets[col][q] == X])
        y_score = len([col for col in columns_to_use if pp.pauli_gadgets[col][q] == Y])
        z_score = len([col for col in columns_to_use if pp.pauli_gadgets[col][q] == Z])
        # score = np.mean([i_score, x_score, y_score, z_score])
        # score = max([i_score, x_score, y_score, z_score]) - \
        #         min([i_score, x_score, y_score, z_score])
        score = max([i_score, x_score, y_score, z_score]) - min(
            [i_score, x_score, y_score, z_score]
        )
        qubit_scores.append((q, score))
    return max(qubit_scores, key=lambda x: x[1])[0]


def update_gadget_single_column(
    pp: PauliPolynomial, qc: PauliCircuit, q: int, p: Pauli, columns_to_use
):
    if p == X:
        gate = H(q)
        pp.propagate(gate, columns_to_use)
        qc.h(q)
    elif p == Y:
        gate = V(q)
        pp.propagate(gate, columns_to_use)
        qc.v(q)
    elif p == Z:
        pass  # Nothing to do here
    else:
        raise ValueError("Invalid Pauli")


def find_common_paulis(q, pp: PauliPolynomial, columns_to_use):
    common_paulis = []
    for col in columns_to_use:
        if pp[col][q] != I:
            common_paulis.append(pp[col][q])
    common_paulis = list(set(common_paulis))
    if len(common_paulis) == 1:
        return common_paulis[0]
    return None


def update_single_qubits(
    pp: PauliPolynomial, c: PauliCircuit, qubits: list, columns_to_use
):
    change = False
    for q in qubits:
        p = find_common_paulis(q, pp, columns_to_use)
        if p is not None:
            update_gadget_single_column(pp, c, q, p, columns_to_use)
            change = True
    return change


def is_compatible(pp: PauliPolynomial, q1, q2, columns_to_use):
    return find_compatible_pair(pp, q1, q2, columns_to_use) is not None


def find_compatible_pair(pp: PauliPolynomial, q1, q2, columns_to_use):
    for p1 in [X, Y, Z]:
        for p2 in [X, Y, Z]:
            found_pair = True
            for l in columns_to_use:
                p_gdt = pp[l][q1]
                p_gdt2 = pp[l][q2]
                a_valid = p_gdt in [I, p1]
                b_valid = p_gdt2 in [I, p2]
                if a_valid != b_valid:
                    found_pair = False
                    break
            if found_pair:
                return p1, p2

    return None


def pick_best_pair(pp, G: nx.Graph, columns_to_use, qubits):
    for q_1, q_2 in G.edges:
        if q_1 in qubits and q_2 in qubits:
            pairs = find_compatible_pair(pp, q_1, q_2, columns_to_use)
            if pairs is not None:
                return pairs, (q_1, q_2)
    return None


def filter_identity_qubits(pp: PauliPolynomial, qubits, columns_to_use):
    non_identity_qubits = []
    for q in qubits:
        if pp[columns_to_use[0]][q] != I:
            non_identity_qubits.append(q)
    return non_identity_qubits


def update_pair_qubits(
    pp: PauliPolynomial, c: PauliCircuit, topology, qubits, columns_to_use
):
    non_visited_qubits = [q for q in qubits]
    non_visited_qubits = filter_identity_qubits(pp, non_visited_qubits, columns_to_use)
    qubit_pairs = pick_best_pair(
        pp,
        topology.to_nx.subgraph(non_visited_qubits),
        columns_to_use,
        non_visited_qubits,
    )
    (p1, p2), (q_1, q_2) = qubit_pairs
    non_visited_qubits.remove(q_1)
    non_visited_qubits.remove(q_2)
    update_gadget_single_column(pp, c, q_1, p1, columns_to_use)
    update_gadget_single_column(pp, c, q_2, p2, columns_to_use)

    pp.propagate(CX(q_1, q_2), columns_to_use)
    c.cx(q_1, q_2)


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
    cols = []
    if not col_x:
        return col_i, X, Z, col_y, Z, col_z
    if not col_y:
        return col_i, X, Y, col_x, Z, col_z

    if len(col_x) != 0 and len(col_y) != 0:
        cols.append((X, Y, col_x + col_y, Z, col_z, len(col_x) + len(col_y)))
    if len(col_x) != 0 and len(col_z) != 0:
        cols.append((X, Z, col_x + col_z, Y, col_y, len(col_x) + len(col_z)))
    if len(col_y) != 0 and len(col_z):
        cols.append((Y, Z, col_y + col_z, X, col_x, len(col_y) + len(col_z)))

    if cols:
        type_two_1, type_two_2, cols_2, type_col1, col1, _ = max(
            cols, key=lambda x: x[-1]
        )
    else:
        raise Exception("Invalid State")
    return col_i, type_two_1, type_two_2, cols_2, type_col1, col1


def pauli_polynomial_steiner_gray_nc(pp: PauliPolynomial, topo: Topology):
    perm_gadgets = []
    permutation = {k: k for k in range(pp.num_qubits)}
    G = topo.to_nx

    def identity_recurse(columns_to_use, qubits_to_use):
        if not columns_to_use or not qubits_to_use:
            return PauliCircuit(pp.num_qubits)

        G_ = G.subgraph(qubits_to_use)
        non_cutting = [q for q in qubits_to_use if not is_cutting(q, G_)]
        row = pick_row(pp, columns_to_use, non_cutting)

        pp_i, pp_x, pp_y, pp_z = partition_pauli_polynomial(pp, row, columns_to_use)

        remaining_qubits = [q for q in qubits_to_use if q != row]

        qc_i = identity_recurse(pp_i, remaining_qubits)
        qc_x = p_recurse(pp_x, qubits_to_use, row, X)
        qc_y = p_recurse(pp_y, qubits_to_use, row, Y)
        qc_z = p_recurse(pp_z, qubits_to_use, row, Z)

        qc_out = PauliCircuit(pp.num_qubits)
        qc_out += qc_i
        qc_out += qc_x
        qc_out += qc_y
        qc_out += qc_z
        return qc_out

    def p_recurse(columns_to_use, qubits_to_use, row, rec_type):
        assert rec_type in [X, Y, Z]
        if not columns_to_use or not qubits_to_use:
            return PauliCircuit(pp.num_qubits)
        G_: nx.Graph = G.subgraph(qubits_to_use)
        neighbours = [q for q in G_.neighbors(row)]
        assert neighbours, "No neighbours found"

        row_next = pick_row(pp, columns_to_use, neighbours)

        pp_i, t_pp2_1, tpp2_2, pp2, t_p1, pp1 = partition_pauli_polynomial_(
            pp, row_next, columns_to_use
        )

        remaining_qubits = [q for q in qubits_to_use if q != row_next]
        if not is_cutting(row_next, G_):
            qc_i = identity_recurse(pp_i, remaining_qubits)
        else:
            qc_i = check_identity(pp_i, qubits_to_use, row, row_next, rec_type)

        qc_two = simplify_twp_p(
            pp2, qubits_to_use, row, rec_type, row_next, t_pp2_1, tpp2_2
        )
        qc_one = simplify_one_p(pp1, qubits_to_use, row, rec_type, row_next, t_p1)

        qc_out = PauliCircuit(pp.num_qubits)
        qc_out += qc_i
        qc_out += qc_two
        qc_out += qc_one
        return qc_out

    def check_columns(columns_to_use):
        qc = PauliCircuit(pp.num_qubits)
        to_remove = []
        for col in columns_to_use:
            if pp[col].num_legs() == 1:
                row = [q for q in range(pp.num_qubits) if pp[col][q] != I][0]
                col_type = pp[col][row]

                if col_type == X:
                    qc.h(row)
                elif col_type == Y:
                    qc.v(row)

                qc.rz(pp.pauli_gadgets[col].angle, row)
                perm_gadgets.append(col)
                to_remove.append(col)

                if col_type == X:
                    qc.h(row)
                elif col_type == Y:
                    qc.vdg(row)
        for col in to_remove:
            columns_to_use.remove(col)
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

        qc_i = identity_recurse(columns_to_use, remaining_qubits)
        qc += qc_i

        qc.cx(row, row_next)
        qc.cx(row_next, row)
        if rec_type == X:
            qc.h(row)
        elif rec_type == Y:
            qc.v(row)
        return qc

    def simplify_twp_p(
        columns_to_use,
        qubits_to_use,
        row,
        rec_type,
        row_next,
        rec_type_next_1,
        rec_type_next_2,
    ):
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
            qc.s(row_next)
            pp.propagate(Sdg(row_next), columns_to_use)
        elif rec_type_next_1 == Y and rec_type_next_2 == Z:
            pass  # just to note this case

        qc.cx(row, row_next)
        pp.propagate(CX(row, row_next), columns_to_use)
        qc += check_columns(columns_to_use)

        columns_to_use_1 = [col for col in columns_to_use if pp[col][row_next] == Z]
        columns_to_use_2 = [col for col in columns_to_use if pp[col][row_next] == Y]
        qc_z = p_recurse(columns_to_use_1, qubits_to_use, row_next, Z)
        qc_y = p_recurse(columns_to_use_2, qubits_to_use, row_next, Y)
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
            qc.sdg(row_next)
        return qc

    def simplify_one_p(
        columns_to_use, qubits_to_use, row, rec_type, row_next, rec_type_next
    ):
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
        qc += check_columns(columns_to_use)

        qc_ = p_recurse(columns_to_use, qubits_to_use, row_next, Z)

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

    circ_out = PauliCircuit(pp.num_qubits)
    columns_to_use = list(range(pp.num_gadgets))
    circ_out += check_columns(columns_to_use)
    circ_out += identity_recurse(columns_to_use, list(range(pp.num_qubits)))

    permutation = [permutation[i] for i in range(pp.num_qubits)]
    return circ_out, perm_gadgets, permutation
