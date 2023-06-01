import unittest

from qiskit import QuantumCircuit

from pauliopt.pauli.clifford_gates import generate_random_clifford, \
    CliffordType, clifford_to_qiskit
from pauliopt.topologies import Topology
from tests.utils import generate_all_combination_pauli_polynomial, \
    pauli_poly_to_tket, verify_equality, check_matching_architecture, get_two_qubit_count


class TestPauliConversion(unittest.TestCase):
    def test_circuit_construction(self):
        """
        Checks in this Unit test:
        1) If one constructs the Pauli Polynomial with our libary the circuits should match the ones of tket
        2) When synthesizing onto a different architecture the circuits should match the ones of tket
        3) Check that our to_qiskit method exports the Pauli Polynomial according to an architecture
        """
        for num_qubits in [2, 3, 4]:
            for topo_creation in [Topology.line, Topology.complete]:
                pp = generate_all_combination_pauli_polynomial(n_qubits=num_qubits)

                topology = topo_creation(pp.num_qubits)
                tket_pp = pauli_poly_to_tket(pp)
                our_synth = pp.to_qiskit(topology)
                self.assertTrue(verify_equality(tket_pp, our_synth),
                                "The resulting Quantum Circuits were not equivalent")
                self.assertTrue(check_matching_architecture(our_synth, topology.to_nx),
                                "The Pauli Polynomial did not match the architecture")
                self.assertEqual(get_two_qubit_count(our_synth),
                                 pp.two_qubit_count(topology),
                                 "Two qubit count needs to be equivalent to to two qubit count of the circuit")

    def test_gate_propagation(self):
        """
        Checks if the clifford Propagation rules are sound for 2, 3, 4 qubits
        """
        for num_qubits in [2, 3, 4]:
            pp = generate_all_combination_pauli_polynomial(n_qubits=num_qubits)
            inital_qc = pp.to_qiskit()
            for gate_class in [CliffordType.CX, CliffordType.CY, CliffordType.CZ,
                               CliffordType.H, CliffordType.S, CliffordType.V]:
                gate = generate_random_clifford(gate_class, num_qubits)
                print(gate_class)
                pp_ = pp.copy().propagate(gate)
                qc = QuantumCircuit(num_qubits)
                qc.compose(clifford_to_qiskit(gate).inverse(), inplace=True)
                qc.compose(pp_.to_qiskit(), inplace=True)
                qc.compose(clifford_to_qiskit(gate), inplace=True)
                self.assertTrue(verify_equality(inital_qc, qc),
                                "The resulting Quantum Circuits were not equivalent")
