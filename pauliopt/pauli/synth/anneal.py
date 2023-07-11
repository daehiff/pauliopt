import numpy as np

from pauliopt.pauli.clifford_gates import CliffordGate
from pauliopt.pauli.clifford_gates import CliffordType, \
    generate_two_qubit_clifford
from pauliopt.pauli.clifford_tableau import CliffordTableau
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.phase.optimized_circuits import _validate_temp_schedule


def pick_random_gate(num_qubits):
    control = np.random.choice([q for q in range(num_qubits)])
    target = np.random.choice([q for q in range(num_qubits) if q != control])
    return control, target


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


def count_legs(pp: PauliPolynomial, gate: CliffordGate):
    pp_ = pp.copy()
    pp_.propagate(gate)
    return pp_.num_legs() - pp.num_legs()


def anneal(pp: PauliPolynomial, topology, schedule=("geometric", 1.0, 0.1),
           nr_iterations=100, gate_set=None):
    if gate_set is None:
        gate_set = [CliffordType.CX, CliffordType.CY, CliffordType.CZ, CliffordType.CXH]

    leg_cache = {}
    clifford_region = CliffordTableau(n_qubits=pp.num_qubits)

    schedule = _validate_temp_schedule(schedule)
    random_nrs = np.random.uniform(0.0, 1.0, size=(nr_iterations,))
    for it in range(nr_iterations):
        t = schedule(it, nr_iterations)
        ctrl, trg = pick_random_gate(pp.num_qubits)
        gate, effect = get_best_gate(pp, ctrl, trg, gate_set, topology,
                                     leg_cache=leg_cache)
        accept_step = effect < 0 or random_nrs[it] < np.exp(-np.log(2) * effect / t)
        if accept_step:
            clifford_region.append_gate(gate)
            pp.propagate(gate)
    try:
        from qiskit import QuantumCircuit
    except:
        raise Exception("Please install qiskit to export the circuit")
    clifford_circ, _ = clifford_region.to_cifford_circuit_arch_aware(topology,
                                                                     include_swaps=False)
    pp_circuit = pp.to_qiskit(topology)

    qc = QuantumCircuit(pp.num_qubits)
    qc.compose(clifford_circ, inplace=True)
    qc.compose(pp_circuit, inplace=True)
    qc.compose(clifford_circ.inverse(), inplace=True)

    perm = list(range(pp.num_qubits))
    gadget_perm = list(range(pp.num_gadgets))
    return qc, gadget_perm, perm
