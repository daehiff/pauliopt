import unittest
import warnings

import numpy as np
import stim
from qiskit import QuantumCircuit
import networkx as nx
from qiskit.providers.fake_provider import FakeLima, FakeLagos

from pauliopt.pauli.clifford_gates import H, S, V, CX, CY, CZ
from pauliopt.pauli.clifford_tableau import CliffordTableau
from pauliopt.pauli.utils import apply_permutation
from pauliopt.topologies import Topology


def verify_equality(qc_in, qc_out):
    """
    Verify the equality up to a global phase
    :param qc_in:
    :param qc_out:
    :return:
    """
    try:
        from qiskit.quantum_info import Operator
    except:
        raise Exception("Please install qiskit to compare to quantum circuits")
    return Operator.from_circuit(qc_in) \
        .equiv(Operator.from_circuit(qc_out))


def reconstruct_tableau(tableau: stim.Tableau):
    xsx, xsz, zsx, zsz, x_signs, z_signs = tableau.to_numpy()
    x_row = np.concatenate([xsx, xsz], axis=1)
    z_row = np.concatenate([zsx, zsz], axis=1)
    return np.concatenate([x_row, z_row], axis=0).astype(np.int64)


def reconstruct_tableau_signs(tableau: stim.Tableau):
    xsx, xsz, zsx, zsz, x_signs, z_signs = tableau.to_numpy()
    signs = np.concatenate([x_signs, z_signs]).astype(np.int64)
    return signs


def circuit_to_tableau(circ: QuantumCircuit):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    tableau = stim.Tableau(circ.num_qubits)
    cnot = stim.Tableau.from_named_gate("CX")
    had = stim.Tableau.from_named_gate("H")
    s = stim.Tableau.from_named_gate("S")
    for op in circ:
        if op.operation.name == "h":
            tableau.append(had, [op.qubits[0].index])
        elif op.operation.name == "s":
            tableau.append(s, [op.qubits[0].index])
        elif op.operation.name == "cx":
            tableau.append(cnot, [op.qubits[0].index, op.qubits[1].index])
        else:
            raise Exception("Unknown operation")
    return tableau


def random_clifford_circuit(nr_gates=20, nr_qubits=4, gate_choice=None):
    qc = QuantumCircuit(nr_qubits)
    if gate_choice is None:
        gate_choice = ["CY", "CZ", "CX", "H", "S", "V"]
    for _ in range(nr_gates):
        gate_t = np.random.choice(gate_choice)
        if gate_t == "CX":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cx(control, target)
        elif gate_t == "CY":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cy(control, target)
        elif gate_t == "CZ":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cz(control, target)
        elif gate_t == "H":
            qubit = np.random.choice([i for i in range(nr_qubits)])
            qc.h(qubit)
        elif gate_t == "S":
            qubit = np.random.choice([i for i in range(nr_qubits)])
            qc.s(qubit)
        elif gate_t == "V":
            qubit = np.random.choice([i for i in range(nr_qubits)])
            qc.sx(qubit)
        elif gate_t == "CX":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cx(control, target)
    return qc


def random_hscx_circuit(nr_gates=20, nr_qubits=4):
    gate_choice = ["H", "S", "CX"]
    return random_clifford_circuit(nr_gates=nr_gates,
                                   nr_qubits=nr_qubits,
                                   gate_choice=gate_choice)


def parse_stim_to_qiskit(circ: stim.Circuit):
    qc = QuantumCircuit(circ.num_qubits)
    for gate in circ:
        if gate.name == "CX":
            targets = [target.value for target in gate.targets_copy()]
            targets = [(targets[i], targets[i + 1]) for i in range(0, len(targets), 2)]
            for (ctrl, target) in targets:
                qc.cx(ctrl, target)
        elif gate.name == "H":
            targets = [target.value for target in gate.targets_copy()]
            for qubit in targets:
                qc.h(qubit)
        elif gate.name == "S":
            targets = [target.value for target in gate.targets_copy()]
            for qubit in targets:
                qc.s(qubit)
        else:
            raise TypeError(f"Unknown Name: {gate.name}")
    return qc


def random_hscx_circuit(nr_gates=20, nr_qubits=4):
    gate_choice = ["H", "S", "CX"]
    qc = QuantumCircuit(nr_qubits)
    for _ in range(nr_gates):
        gate_t = np.random.choice(gate_choice)
        if gate_t == "H":
            qubit = np.random.choice([i for i in range(nr_qubits)])
            qc.h(qubit)
        elif gate_t == "S":
            qubit = np.random.choice([i for i in range(nr_qubits)])
            qc.s(qubit)
        elif gate_t == "CX":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cx(control, target)
    return qc


def check_matching_architecture(qc: QuantumCircuit, G: nx.Graph):
    for gate in qc:
        if gate.operation.num_qubits == 2:
            ctrl, target = gate.qubits
            ctrl, target = ctrl._index, target._index  # TODO refactor this to a non deprecated way
            if not G.has_edge(ctrl, target):
                return False
    return True


class TestCliffordTableau(unittest.TestCase):
    def test_circuit_construction(self):
        for _ in range(10):
            for n_qubits in [4, 8]:
                circ = random_hscx_circuit(nr_gates=800, nr_qubits=n_qubits)
                tableau = circuit_to_tableau(circ)
                circ_stim = parse_stim_to_qiskit(tableau.to_circuit(method="elimination"))

                ct = CliffordTableau.from_circuit(circ)
                circ_out = ct.to_clifford_circuit()
                self.assertTrue(np.allclose(reconstruct_tableau(tableau), ct.tableau),
                                "The Instructions resulted in an incorrect Tableau")
                self.assertTrue(np.allclose(reconstruct_tableau_signs(tableau), ct.signs),
                                "")
                self.assertTrue(verify_equality(circ, circ_stim),
                                "The STIM Circuit resulted in a different circuit")

                self.assertTrue(verify_equality(circ, circ_out),
                                "The resulting circuit from the clifford tableau did not match")

    def test_tableau_synthesis_structured_architectures(self):
        for topo in [Topology.line(5), Topology.line(8),
                     Topology.cycle(5), Topology.cycle(8),
                     Topology.periodic_grid(2, 3), Topology.periodic_grid(2, 4)]:
            for num_gates in [200, 400, 800]:
                circ = random_hscx_circuit(nr_gates=num_gates, nr_qubits=topo.num_qubits)

                ct = CliffordTableau.from_circuit(circ)
                circ_out, perm = ct.to_clifford_circuit_arch_aware_qiskit(topo)

                self.assertTrue(verify_equality(circ, circ_out),
                                "The resulting circuit from the clifford tableau did not match")

                circ_out = apply_permutation(circ_out, perm)
                self.assertTrue(check_matching_architecture(circ_out, topo.to_nx),
                                "The resulting circuit did no match the architecture")

    def test_gate_appending(self):
        for nr_qubits in [4, 8]:
            qc = random_clifford_circuit(nr_gates=2000, nr_qubits=nr_qubits)
            ct = CliffordTableau(n_qubits=qc.num_qubits)
            operations = set()

            for op in qc:
                operations.add(op.operation.name)
                if op.operation.name == "h":
                    gate = H(op.qubits[0].index)
                elif op.operation.name == "s":
                    gate = S(op.qubits[0].index)
                elif op.operation.name == "sx":
                    gate = V(op.qubits[0].index)
                elif op.operation.name == "cx":
                    gate = CX(op.qubits[0].index, op.qubits[1].index)
                elif op.operation.name == "cy":
                    gate = CY(op.qubits[0].index, op.qubits[1].index)
                elif op.operation.name == "cz":
                    gate = CZ(op.qubits[0].index, op.qubits[1].index)
                else:
                    raise Exception(f"Unknown Gate: {op.operation.name}")
                ct.append_gate(gate)

            self.assertTrue(verify_equality(qc, ct.to_clifford_circuit()),
                            "Expected the circuit to match the gates")

    def test_tableau_synthesis_fine_grain(self):
        topo = Topology.complete(6)
        for _ in range(100):
            for num_gates in range(1, 30):
                print(num_gates)
                circ = random_hscx_circuit(nr_gates=num_gates, nr_qubits=topo.num_qubits)

                ct = CliffordTableau.from_circuit(circ)
                circ_out, perm = ct.to_clifford_circuit_arch_aware_qiskit(topo)

                self.assertTrue(verify_equality(circ, circ_out),
                                "The resulting circuit from the clifford tableau did not match")

    def test_tableau_synthesis_ibm_backends(self):
        for backend in [FakeLima(), FakeLagos()]:
            topo = Topology.from_qiskit_backend(backend)
            for num_gates in [200, 400, 800]:
                circ = random_hscx_circuit(nr_gates=num_gates, nr_qubits=topo.num_qubits)
                ct = CliffordTableau.from_circuit(circ)
                circ_out, _ = ct.to_clifford_circuit_arch_aware_qiskit(topo,
                                                                       include_swaps=False)

                self.assertTrue(verify_equality(circ, circ_out),
                                "The resulting circuit from the clifford tableau did not match")

                self.assertTrue(check_matching_architecture(circ_out, topo.to_nx),
                                "The resulting circuit did no match the architecture")
