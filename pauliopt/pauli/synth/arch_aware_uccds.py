import networkx as nx
from qiskit import QuantumCircuit

from pauliopt.pauli.clifford_gates import CX
from pauliopt.pauli.clifford_gates import H, V
from pauliopt.pauli.clifford_region import CliffordRegion
from pauliopt.pauli.pauli_gadget import PauliGadget
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.utils import I, X, Y, Z, Pauli
from pauliopt.topologies import Topology


def pick_row(pp: PauliPolynomial, columns_to_use, qubits_to_use):
    row_scores = []
    for q in qubits_to_use:
        zeros = len([1 for g_idx in columns_to_use if pp[g_idx][q] == I])
        ones = len([1 for g_idx in columns_to_use if pp[g_idx][q] != I])
        row_scores.append((q, max(zeros, ones)))
    return max(row_scores, key=lambda x: x[1])[0]


def partition_pauli_polynomial(pp: PauliPolynomial, row: int, columns_to_use: list):
    pp_i = []
    pp_z = []

    for g_idx in columns_to_use:
        if pp[g_idx][row] == I:
            pp_i.append(g_idx)
        else:
            pp_z.append(g_idx)

    return pp_i, pp_z


def choose_target(pp: PauliPolynomial, row: int, columns_to_use: list, qubits: list):
    qubit_scores = []
    for q in qubits:
        if q == row:
            continue
        non_zeros = len([1 for col in columns_to_use if pp[col][q] == Z])
        qubit_scores.append((q, non_zeros))
    return max(qubit_scores, key=lambda x: x[1])[0]


def is_cutting(vertex, g):
    return vertex in nx.articulation_points(g)


def is_zero_column(pp, row_n, columns_to_use):
    for c in columns_to_use:
        if pp[c][row_n] != I:
            return False
    return True


def pauli_polynomial_steiner_gray_synth(pp: PauliPolynomial, topo: Topology):
    remaining_cliffords = CliffordRegion(pp.num_qubits)
    circ_out = QuantumCircuit(pp.num_qubits)
    remaining_columns = list(range(pp.num_gadgets))
    perm_gadgets = []
    G = topo.to_nx

    def place_cnot(control, target):
        circ_out.cx(control, target)
        remaining_cliffords.prepend_gate(CX(control, target))
        pp.propagate(CX(control, target), remaining_columns)

    def reduce_columns(columns_to_use):
        non_reduced_columns = []
        for col in columns_to_use:
            legs = [q for q in range(pp.num_qubits) if pp.pauli_gadgets[col][q] != I]
            if len(legs) == 1:
                q = legs[0]
                circ_out.rz(pp[col].angle.to_qiskit, q)
                perm_gadgets.append(col)
                remaining_columns.remove(col)
            else:
                non_reduced_columns.append(col)
        return non_reduced_columns

    def identity_recurse(columns_to_use, qubits_to_use):
        if not columns_to_use or not qubits_to_use:
            return
        G_ = G.subgraph(qubits_to_use)
        non_cutting = [q for q in qubits_to_use if not is_cutting(q, G_)]
        row = pick_row(pp, columns_to_use, non_cutting)
        pp_i, pp_z = partition_pauli_polynomial(pp, row, columns_to_use)
        identity_recurse(pp_i, [q for q in qubits_to_use if q != row])
        p_recurse(pp_z, qubits_to_use, row, Z)

    def p_recurse(columns_to_use, qubits_to_use, row, rec_type):
        assert rec_type in [X, Y, Z]
        if not columns_to_use or not qubits_to_use:
            return

        G_ = G.subgraph(qubits_to_use)
        neighbours = [q for q in G_.neighbors(row)]
        if not neighbours:
            return
        row_n = choose_target(pp, row, columns_to_use, neighbours)
        if is_zero_column(pp, row_n, columns_to_use):
            place_cnot(row, row_n)
            place_cnot(row_n, row)
        else:
            place_cnot(row, row_n)
            columns_to_use = reduce_columns(columns_to_use)

        pp_i, pp_z = partition_pauli_polynomial(pp, row, columns_to_use)
        identity_recurse(pp_i, [q for q in qubits_to_use if q != row])
        p_recurse(pp_z, qubits_to_use, row, Z)

    columns = list(range(pp.num_gadgets))
    columns = reduce_columns(columns)
    identity_recurse(columns, list(range(pp.num_qubits)))

    # TODO test rec steiner gauss vs tableau synthesis
    # circ_out.compose(remaining_cliffords.to_qiskit(method="ct_resynthesis",
    #                                                topology=topo)[0],
    #                  inplace=True)
    return circ_out, perm_gadgets, remaining_cliffords


def compare(pp, col1, col2):
    alph_order = {I: 0, X: 1, Y: 2, Z: 3}
    for q in range(pp.num_qubits):
        if alph_order[pp[col1][q]] > alph_order[pp[col2][q]]:
            return True
        elif alph_order[pp[col1][q]] < alph_order[pp[col2][q]]:
            return False

    return False


def sort_pp_region_alphabetical(pp: PauliPolynomial, columns):
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if compare(pp, columns[i], columns[j]):
                columns[i], columns[j] = columns[j], columns[i]


def sequence_pauli_polynomial(pp: PauliPolynomial):
    G = nx.Graph()
    G.add_nodes_from(list(range(pp.num_gadgets)))
    for idx, gadget in enumerate(pp.pauli_gadgets):
        for idx2, gadget2 in enumerate(pp.pauli_gadgets):
            if idx != idx2:
                if not gadget.commutes(gadget2):
                    G.add_edge(idx, idx2)
    d = nx.coloring.greedy_color(G)
    colors = {}
    for idx, color in d.items():
        if color not in colors:
            colors[color] = []
        colors[color].append(idx)

    colors_set = []
    for _, v in colors.items():
        sort_pp_region_alphabetical(pp, v)
        colors_set.append((v, len(v)))

    colors_set = sorted(colors_set, key=lambda x: x[1])
    colors_set = [v for v, _ in colors_set]

    return colors_set


def find_common_paulis(q, pp: PauliPolynomial):
    common_paulis = []
    for idx, gadget in enumerate(pp.pauli_gadgets):
        if gadget.paulis[q] != I:
            common_paulis.append(gadget.paulis[q])
    common_paulis = list(set(common_paulis))
    if len(common_paulis) == 1:
        return common_paulis[0]
    return None


def update_gadget_single_column(pp: PauliPolynomial, c: CliffordRegion, q: int, p: Pauli):
    if p == X:
        gate = H(q)
        pp.propagate(gate)
        c.append_gate(gate)
    elif p == Y:
        gate = V(q)
        pp.propagate(gate)
        c.append_gate(gate)
    elif p == Z:
        pass  # Nothing to do here
    else:
        raise ValueError("Invalid Pauli")


def update_single_qubits(pp: PauliPolynomial, c: CliffordRegion, qubits: list):
    change = False
    for q in qubits:
        p = find_common_paulis(q, pp)
        if p is not None:
            update_gadget_single_column(pp, c, q, p)
            qubits.remove(q)
            change = True
    return change


def find_compatible_pair(pp: PauliPolynomial, q1, q2):
    for p1 in [X, Y, Z]:
        for p2 in [X, Y, Z]:
            found_pair = True
            for l in range(pp.num_gadgets):
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


def pick_best_pair(qubits, pp, topology):
    pair_list = []
    for q_1 in qubits:
        for q_2 in [q for q in qubits if q != q_1]:
            pairs = find_compatible_pair(pp, q_1, q_2)
            if pairs is not None:
                pair_list.append(((pairs, (q_1, q_2)), topology.dist(q_1, q_2)))
    if len(pair_list) == 0:
        return None
    else:
        return min(pair_list, key=lambda x: x[1])[0]


def update_pair_qubits(pp: PauliPolynomial, c: CliffordRegion, qubits: list, topology):
    qubit_pairs = pick_best_pair(qubits, pp, topology)
    if qubit_pairs:
        (p1, p2), (q_1, q_2) = qubit_pairs
        update_gadget_single_column(pp, c, q_1, p1)
        update_gadget_single_column(pp, c, q_2, p2)
        cx = CX(q_1, q_2)
        pp.propagate(cx)
        c.append_gate(cx)
        qubits.remove(q_2)
        return True
    return False


def is_in_qubits(gadget: PauliGadget, qubits: list):
    for q in qubits:
        if gadget[q] != I:
            return True
    return False


def update_greedy(pp: PauliPolynomial, c: CliffordRegion, qubits: list):
    all_legs = [(l, pp[l].num_legs()) for l in range(pp.num_gadgets)
                if is_in_qubits(pp[l], qubits)]
    l = min(all_legs, key=lambda x: x[1])[0]
    q0 = [q for q in qubits if pp[l][q] != I][0]
    decomposition, _ = pp.pauli_gadgets[l].decompose(q0=q0)
    for gate in decomposition:
        pp = pp.propagate(gate)
        c.append_gate(gate)

    qubits.remove(q0)
    return pp


def is_i_column(q, pp):
    for gadget in pp.pauli_gadgets:
        if gadget[q] != I:
            return False
    return True


def diagonalize_pauli_polynomial(pp: PauliPolynomial, topology: Topology):
    qubits = list(range(pp.num_qubits))
    c = CliffordRegion(pp.num_qubits)

    while qubits:
        # filter out all qubits with pure I values
        qubits = [q for q in qubits if not is_i_column(q, pp)]

        update_single_qubits(pp, c, qubits)

        if not qubits:
            break
        len_prev = len(qubits)
        update_pair_qubits(pp, c, qubits, topology)

        if len_prev == len(qubits):
            pp = update_greedy(pp, c, qubits)

    c_ = CliffordRegion(c.num_qubits)
    for gate in reversed(c.gates):
        c_.append_gate(gate)

    return pp, c_


def is_z_phase_poly(pp: PauliPolynomial):
    for l in range(pp.num_gadgets):
        for q in range(pp.num_qubits):
            if pp[l][q] not in [I, Z]:
                return False
    return True


def uccds_synthesis(pp: PauliPolynomial, topo: Topology):
    remaining_cols = list(range(pp.num_gadgets))
    sub_regions = sequence_pauli_polynomial(pp)
    qc = QuantumCircuit(pp.num_qubits)
    gadget_order = []

    global_cliffords = CliffordRegion(pp.num_qubits)

    for region in sub_regions:
        remaining_cols = list(filter(lambda x: x not in region, remaining_cols))
        pp_sub = PauliPolynomial(pp.num_qubits)
        pp_sub.pauli_gadgets = [pp.pauli_gadgets[l].copy() for l in region]
        pp_sub, c_diag = diagonalize_pauli_polynomial(pp_sub, topo)
        c_circ, _ = c_diag.to_qiskit("ct_resynthesis", topology=topo)
        if not is_z_phase_poly(pp_sub):
            raise Exception("Not a Z-phase polynomial")

        sub_circ, sub_gadget_perm, rem_cliff = \
            pauli_polynomial_steiner_gray_synth(pp_sub, topo)

        for gate in reversed(c_diag.gates):
            global_cliffords.prepend_gate(gate)
            pp = pp.propagate(gate)

        for gate in reversed(rem_cliff.gates):
            global_cliffords.prepend_gate(gate)
            pp = pp.propagate(gate)

        qc.compose(c_circ.inverse(), inplace=True)
        qc.compose(sub_circ, inplace=True)
        gadget_order += [region[i] for i in sub_gadget_perm]

    qc.compose(global_cliffords.to_qiskit("ct_resynthesis", topology=topo)[0],
               inplace=True)
    perm = list(range(pp.num_qubits))
    return qc, gadget_order, perm
