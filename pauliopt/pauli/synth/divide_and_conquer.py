import numpy as np

from pauliopt.pauli.clifford_gates import CliffordGate, CliffordType, \
    generate_random_clifford, generate_two_qubit_clifford
from pauliopt.pauli.clifford_region import CliffordRegion
from pauliopt.pauli.pauli_circuit import PauliCircuit
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.phase.optimized_circuits import _validate_temp_schedule
from pauliopt.topologies import Topology
from pauliopt.pauli.clifford_gates import CX, CY, CZ, CliffordGate, ControlGate
from pauliopt.pauli.clifford_region import CliffordRegion
from pauliopt.pauli.clifford_tableau import CliffordTableau
from pauliopt.pauli.pauli_gadget import PauliGadget
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.utils import apply_permutation
from pauliopt.phase.optimized_circuits import _validate_temp_schedule
from pauliopt.topologies import Topology
from pauliopt.pauli.utils import X, Y, Z, I

DESIRED_NEIGHBORS = [(X, X), (Y, X), (Z, Z), (Z, Y)]
UNDESIRABLE_NEIGHBORS = [(X, I), (Y, I), (Z, I)]


def compute_effect(pp, gate, topology, leg_cache=None):
    pp_ = pp.copy()
    pp_.propagate(gate)
    return pp_.two_qubit_count(topology, leg_cache=leg_cache) - \
           pp.two_qubit_count(topology, leg_cache=leg_cache)


def get_best_gate(pp, c, t, gate_set, topology, leg_cache=None):
    gate_scores = []
    for gate in gate_set:
        gate = generate_two_qubit_clifford(gate, c, t)
        effect = compute_effect(pp, gate, topology, leg_cache=leg_cache)
        gate_scores.append((gate, effect))
    return min(gate_scores, key=lambda x: x[1])


def compute_global_permutation(pp: PauliPolynomial, topology: Topology):
    import networkx as nx
    G_match = nx.Graph()
    for e1 in range(pp.num_qubits):
        for e2 in range(pp.num_qubits):
            if e1 != e2:
                G_match.add_edge(e1, e2, weight=0)
    G_match.add_nodes_from(range(pp.num_qubits))
    for gadget in pp.pauli_gadgets:
        assert isinstance(gadget, PauliGadget)
        for e1 in range(pp.num_qubits):
            for e2 in range(pp.num_qubits):
                if e1 != e2:
                    p1 = gadget.paulis[e1]
                    p2 = gadget.paulis[e2]
                    if (p1, p2) in DESIRED_NEIGHBORS or (p2, p1) in DESIRED_NEIGHBORS:
                        G_match[e1][e2]['weight'] += topology.dist(e1, e2)
                    elif (p1, p2) in UNDESIRABLE_NEIGHBORS or (
                            p2, p1) in UNDESIRABLE_NEIGHBORS:
                        G_match[e1][e2]['weight'] -= topology.dist(e1, e2)
    return dict(nx.maximal_matching(G_match))


def optimize_pauli_polynomial(c_l: CliffordTableau, pp: PauliPolynomial,
                              c_r: CliffordTableau, topology: Topology,
                              gate_set=None, leg_cache=None):
    if gate_set is None:
        gate_set = [CliffordType.CX,
                    CliffordType.CY,
                    CliffordType.CZ, CliffordType.CXH]

    for c in range(pp.num_qubits):
        for t in range(pp.num_qubits):
            if c != t:
                for _ in range(len(gate_set)):
                    gate, effect = get_best_gate(pp, c, t, gate_set, topology,
                                                 leg_cache=leg_cache)
                    dist = topology.dist(c, t)
                    if effect + 2 * dist <= 0:
                        pp = pp.propagate(gate)
                        c_l.append_gate(gate)
                        c_r.prepend_gate(gate)

    return c_l, pp, c_r


def compare(pp: PauliPolynomial, prev, now, next):
    return pp.commutes(now, next) and \
           (pp.mutual_legs(prev, now) < pp.mutual_legs(prev, next))


def sort_pauli_polynomial(pp: PauliPolynomial):
    col_idx = 1
    while col_idx < pp.num_gadgets - 1:
        prev_col_idx = col_idx - 1
        col_idx_ = col_idx
        new_col_idx = col_idx
        while new_col_idx < pp.num_gadgets and compare(pp, prev_col_idx,
                                                       col_idx_, new_col_idx):
            pp.swap_gadgets(col_idx_, new_col_idx)
            prev_col_idx = col_idx_
            col_idx_ = new_col_idx
            new_col_idx = col_idx_ + 1
        col_idx += 1
    return pp


def split_pauli_polynomial(pp: PauliPolynomial):
    pp_left = PauliPolynomial(pp.num_qubits)
    pp_right = PauliPolynomial(pp.num_qubits)

    for gadget in pp.pauli_gadgets[:pp.num_gadgets // 2]:
        pp_left >>= gadget

    for gadget in pp.pauli_gadgets[pp.num_gadgets // 2:]:
        pp_right >>= gadget

    return pp_left, pp_right


def divide_and_conquer(pp: PauliPolynomial, topology: Topology):
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        raise ImportError("Qiskit must be installed to use synth_divide_and_conquer")

    permutation = compute_global_permutation(pp, topology)
    gadget_perm = list(range(pp.num_gadgets))
    pp.permute(permutation)

    c_l = CliffordTableau(pp.num_qubits)
    c_r = CliffordTableau(pp.num_qubits)
    legs_cache = {}
    regions = synth_divide_and_conquer_(c_l, pp, c_r, topology, leg_cache=legs_cache)

    circ_out = PauliCircuit(pp.num_qubits)
    for region in regions:
        if isinstance(region, PauliPolynomial):
            circ_out += region.to_circuit(topology=topology)
        else:
            circ, _ = region.to_cifford_circuit_arch_aware(topology, include_swaps=False)
            circ_out += circ

    perm = list(range(pp.num_qubits))
    for i, j in permutation.items():
        perm[i], perm[j] = perm[j], perm[i]
    return circ_out, gadget_perm, perm


def synth_divide_and_conquer_(c_l: CliffordTableau, pp: PauliPolynomial,
                              c_r: CliffordTableau, topology: Topology,
                              leg_cache=None):
    c_l, pp, c_r = optimize_pauli_polynomial(c_l, pp, c_r, topology, leg_cache=leg_cache)
    if pp.num_gadgets <= 2:
        return [c_l, pp, c_r]

    c_center = CliffordTableau(pp.num_qubits)
    pp = sort_pauli_polynomial(pp)
    pp_left, pp_right = split_pauli_polynomial(pp)
    regions_left = synth_divide_and_conquer_(c_l, pp_left, c_center, topology,
                                             leg_cache=leg_cache)
    regions_right = synth_divide_and_conquer_(c_center, pp_right, c_r, topology,
                                              leg_cache=leg_cache)

    return regions_left[:-1] + [c_center] + regions_right[1:]
