import numpy as np

from pauliopt.pauli.clifford_gates import CliffordGate, CliffordType, \
    generate_random_clifford
from pauliopt.pauli.clifford_region import CliffordRegion
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.phase.optimized_circuits import _validate_temp_schedule
from pauliopt.topologies import Topology


def pick_random_gate(num_qubits, gate_set=None):
    """
    Helper method to pick a random gate from the gate set.
    :param num_qubits: Number of qubits
    :param gate_set: List of gates to pick from
    """
    if gate_set is None:
        gate_set = [CliffordType.CX, CliffordType.CY, CliffordType.CZ]

    gate = np.random.choice(gate_set)

    return generate_random_clifford(gate, num_qubits)


def compute_effect(pp: PauliPolynomial, gate: CliffordGate, topology: Topology,
                   leg_cache=None):
    """
    Helper method to compute the effect of a gate on a Pauli polynomial given a topology.
    :param pp: Pauli polynomial
    :param gate: Clifford gate
    :param topology: Topology
    :param leg_cache: Cache for the legs
    """
    pp_ = pp.copy()
    pp_.propagate(gate)

    return pp_.two_qubit_count(topology, leg_cache=leg_cache) - pp.two_qubit_count(
        topology, leg_cache=leg_cache)


def anneal(pp: PauliPolynomial, topology, schedule=("geometric", 1.0, 0.1),
           nr_iterations=100):
    """
    Synthesize the Pauli polynomial to a circuit using simulated annealing.
    :param pp: Pauli polynomial
    :param topology: Topology
    :param schedule: Schedule for the temperature
            (possible types of scheudles: "geometric", "linear", "exponential")
    :param nr_iterations: Number of iterations
    """
    leg_cache = {}
    clifford_region = CliffordRegion(pp.num_qubits)

    schedule = _validate_temp_schedule(schedule)
    random_nrs = np.random.uniform(0.0, 1.0, size=(nr_iterations,))
    num_qubits = pp.num_qubits
    for it in range(nr_iterations):
        t = schedule(it, nr_iterations)
        gate = pick_random_gate(num_qubits)
        effect = 2 + compute_effect(pp, gate, topology, leg_cache=leg_cache)
        accept_step = effect < 0 or random_nrs[it] < np.exp(-np.log(2) * effect / t)
        if accept_step:
            clifford_region.add_gate(gate)  # TODO optimize clifford regions
            pp.propagate(gate)
    try:
        from qiskit import QuantumCircuit

    except:
        raise Exception("Please install qiskit to export the circuit")

    qc = QuantumCircuit(pp.num_qubits)
    qc.compose(clifford_region.to_qiskit(), inplace=True)  # TODO route on architecture
    qc.compose(pp.to_qiskit(topology), inplace=True)
    qc.compose(clifford_region.to_qiskit().inverse(), inplace=True)
    return qc
