import networkx as nx
import numpy as np
from networkx import find_cliques
from networkx.algorithms.approximation import max_clique
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from pauliopt.pauli.clifford_gates import CX
from pauliopt.pauli.clifford_gates import H, V
from pauliopt.pauli.clifford_region import CliffordRegion
from pauliopt.pauli.clifford_tableau import steiner_reduce_column, relabel_graph_inplace, \
    pick_pivots, CliffordTableau
from pauliopt.pauli.pauli_gadget import PauliGadget
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.utils import I, X, Y, Z, Pauli
from pauliopt.topologies import Topology


def to_cifford_circuit_arch_aware(ct: CliffordTableau, topo: Topology, permutation,
                                  swappable_nodes):
    qc = QuantumCircuit(ct.n_qubits)
    remaining = ct.inverse()
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
        pivot_col, pivot_row = pick_pivots(G, remaining, swappable_nodes, True)

        if is_cutting(pivot_col, G):
            non_cutting_vectices = [(node, nx.shortest_path_length(G, source=node,
                                                                   target=pivot_col,
                                                                   weight="weight"))
                                    for node in G.nodes if not is_cutting(node,
                                                                          G) and node in swappable_nodes]
            non_cutting = min(non_cutting_vectices, key=lambda x: x[1])[0]

            relabel_graph_inplace(G, non_cutting, pivot_col)
            # remaining.swap_cols(parent, child)
            permutation[pivot_col], permutation[non_cutting] = \
                permutation[non_cutting], permutation[pivot_col]

        steiner_reduce_column(pivot_col, G, remaining,
                              apply, swappable_nodes, permutation, True)

        if pivot_col in swappable_nodes:
            swappable_nodes.remove(pivot_col)
        G.remove_node(pivot_col)

    signs_copy_z = remaining.signs[ct.n_qubits:2 * ct.n_qubits].copy()
    for col in range(ct.n_qubits):
        if signs_copy_z[col] != 0:
            apply("H", (col,))

    for col in range(ct.n_qubits):
        if signs_copy_z[col] != 0:
            apply("S", (col,))
            apply("S", (col,))

    for col in range(ct.n_qubits):
        if signs_copy_z[col] != 0:
            apply("H", (col,))

    for col in range(ct.n_qubits):
        if remaining.signs[col] != 0:
            apply("S", (col,))
            apply("S", (col,))
    return qc, permutation


def sequence_pauli_polynomial(pp: PauliPolynomial):
    G = nx.Graph()
    G.add_nodes_from(list(range(pp.num_gadgets)))
    for idx, gadget in enumerate(pp.pauli_gadgets):
        for idx2, gadget2 in enumerate(pp.pauli_gadgets):
            if idx != idx2:
                if gadget.commutes(gadget2):
                    G.add_edge(idx, idx2)
    colors_set = []
    while G.nodes:
        d = max_clique(G)
        colors_set.append(list(d))
        G.remove_nodes_from(d)

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


def update_single_qubits(pp: PauliPolynomial, c: CliffordRegion, qubits: list):
    change = False
    for q in qubits:
        p = find_common_paulis(q, pp)
        if p is not None:
            update_gadget_single_column(pp, c, q, p)
            qubits.remove(q)
            change = True
    return change


def update_pair_qubits(pp: PauliPolynomial, c: CliffordRegion, qubits: list):
    for q_1 in qubits:
        for q_2 in [q for q in qubits if q != q_1]:
            pairs = find_compatible_pair(pp, q_1, q_2)
            if pairs is not None:
                p1, p2 = pairs
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

    # qubits.remove(q0)
    return pp


def is_i_column(q, pp):
    for gadget in pp.pauli_gadgets:
        if gadget[q] != I:
            return False
    return True


def diagonalize_pauli_polynomial(pp: PauliPolynomial):
    qubits = list(range(pp.num_qubits))
    c = CliffordRegion(pp.num_qubits)

    while qubits:
        qubits = [q for q in qubits if not is_i_column(q, pp)]
        # filter out all qubits with pure I values
        update_single_qubits(pp, c, qubits)
        if not qubits:
            break
        len_prev = len(qubits)
        update_pair_qubits(pp, c, qubits)

        if len_prev == len(qubits):
            pp = update_greedy(pp, c, qubits)

            # TODO why?!
    c_ = CliffordRegion(c.num_qubits)
    for gate in reversed(c.gates):
        c_.append_gate(gate)

    return pp, c_


def is_cutting(vertex, g):
    return vertex in nx.articulation_points(g)


def non_cutting(g, qubits):
    g_ = g.subgraph(qubits)
    return [q for q in qubits if not is_cutting(q, g_)]


def choose_control(columns_to_use: list, pp: PauliPolynomial, G: nx.Graph):
    qubit_scores = []
    for q in G.nodes:
        if not is_cutting(q, G):
            non_zeros = len([1 for col in columns_to_use if pp[col][q] != I])
            zeros = len([1 for col in columns_to_use if pp[col][q] == I])
            qubit_scores.append((q, max(non_zeros, zeros)))
    return max(qubit_scores, key=lambda x: x[1])[0]


def choose_target(choosen_row: int, columns_to_use: list,
                  pp: PauliPolynomial, G: nx.Graph):
    qubit_scores = []
    for q in G.nodes:
        if q == choosen_row:
            continue
        if q in G.neighbors(choosen_row):
            non_zeros = len([1 for col in columns_to_use
                             if pp[col][q] != pp[col][choosen_row]])
            qubit_scores.append((q, non_zeros))
    return max(qubit_scores, key=lambda x: x[1])[0]


def split_cols(columns_to_use: list, pp: PauliPolynomial, row: int):
    cols1 = []
    cols0 = []
    for col in columns_to_use:
        if pp[col][row] == I:
            cols0.append(col)
        else:
            cols1.append(col)
    return cols0, cols1


def get_zero_length(pp: PauliPolynomial, columns_to_use: list, row: int, row1: int):
    sum = 0
    for col in columns_to_use:
        if pp[col][row] != I and pp[col][row1] != I:
            sum += 1
    return sum


def ariannes_synth(pp: PauliPolynomial, topology: Topology, swappable_nodes, permutation):
    perm_gadgets = []
    remaining_parities = CliffordRegion(pp.num_qubits)
    circ_out = QuantumCircuit(pp.num_qubits)
    remaining_columns = list(range(pp.num_gadgets))
    G = topology.to_nx

    def place_cnot(control, target):
        circ_out.cx(control, target)
        remaining_parities.prepend_gate(CX(control, target))
        pp.propagate(CX(control, target), remaining_columns)

        if control in swappable_nodes:
            swappable_nodes.remove(control)
        if target in swappable_nodes:
            swappable_nodes.remove(target)

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

    def base_recursion(columns_to_use, qubits_to_use):
        if not columns_to_use or not qubits_to_use:
            return
        G_ = G.subgraph(qubits_to_use)
        chosen_row = choose_control(columns_to_use, pp, G_)

        cols0, cols1 = split_cols(columns_to_use, pp, chosen_row)
        base_recursion(cols0, [q for q in qubits_to_use if q != chosen_row])
        one_recursion(cols1, qubits_to_use, chosen_row)

    def one_recursion(columns_to_use, qubits_to_use, choosen_row):
        if not qubits_to_use or not columns_to_use:
            return
        G_ = G.subgraph(qubits_to_use)
        row_n = choose_target(choosen_row, columns_to_use, pp, G_)

        if get_zero_length(pp, columns_to_use, choosen_row, row_n) == 0:
            print("Staaawp")
            place_cnot(row_n, choosen_row)
            place_cnot(choosen_row, row_n)
        else:
            place_cnot(choosen_row, row_n)
            columns_to_use = reduce_columns(columns_to_use)

        cols0, cols1 = split_cols(columns_to_use, pp, choosen_row)
        base_recursion(cols0, [q for q in qubits_to_use if q != choosen_row])
        one_recursion(cols1, qubits_to_use, choosen_row)

    columns = reduce_columns(list(range(pp.num_gadgets)))
    base_recursion(columns, list(range(pp.num_qubits)))
    return circ_out, remaining_parities, perm_gadgets


def is_diagonal(pp: PauliPolynomial):
    for l in range(pp.num_gadgets):
        for q in range(pp.num_qubits):
            if pp[l][q] != I and pp[l][q] != Z:
                return False
    return True


def apply_qubit_permutation(pp: PauliPolynomial, permutation):
    n = pp.num_qubits
    visited = [False] * n  # Keep track of visited indices

    for i in range(n):
        if not visited[i]:
            visited[i] = True
            visited[permutation[i]] = True
            pp.swap_rows(i, permutation[i])


def synth_cowtan(pp: PauliPolynomial, topology: Topology):
    remaining_cols = list(range(pp.num_gadgets))
    sub_regions = sequence_pauli_polynomial(pp)
    qc = QuantumCircuit(pp.num_qubits)
    gadget_order = []

    permuation = {q: q for q in range(pp.num_qubits)}
    swappable_nodes = list(range(pp.num_qubits))
    global_cliffords = CliffordRegion(pp.num_qubits)

    for region in sub_regions:
        remaining_cols = list(filter(lambda x: x not in region, remaining_cols))
        pp_sub = PauliPolynomial(pp.num_qubits)
        pp_sub.pauli_gadgets = [pp.pauli_gadgets[l].copy() for l in region]
        # apply_qubit_permutation(pp_sub, permuation)

        pp_sub, c_sub = diagonalize_pauli_polynomial(pp_sub)
        if not is_diagonal(pp_sub):
            raise Exception(f"Pauli Polynomial failed to be diagonalized")
        circ_sub, remaining_parities, ariannes_order = \
            ariannes_synth(pp_sub, topology, swappable_nodes, permuation)
        gadget_order += [region[i] for i in ariannes_order]

        # TODO include permutation in whole process

        ct = c_sub.to_tableau()
        # circ_clifford, permuation = \
        #     to_cifford_circuit_arch_aware(ct, topology, permuation, swappable_nodes)
        circ_clifford, _ = c_sub.to_qiskit("ct_resynthesis", topology,
                                           include_swaps=True)

        qc.compose(circ_clifford.inverse(), inplace=True)
        qc.compose(circ_sub, inplace=True)
        # qc.compose(circ_rp, inplace=True)
        # qc.compose(circ_clifford, inplace=True)

        for gate in reversed(c_sub.gates):
            pp = pp.propagate(gate)
            global_cliffords.prepend_gate(gate)

        for gate in reversed(remaining_parities.gates):
            pp.propagate(gate)
            global_cliffords.prepend_gate(gate)

    ct = global_cliffords.to_tableau()
    # c_global, permuation = \
    #     to_cifford_circuit_arch_aware(ct, topology, permuation, swappable_nodes)
    c_global, _ = global_cliffords.to_qiskit("ct_resynthesis", topology,
                                             include_swaps=True)
    qc.compose(c_global, inplace=True)

    print(permuation)
    permuation = [permuation[q] for q in range(pp.num_qubits)]
    print(swappable_nodes)
    return qc, permuation, gadget_order
