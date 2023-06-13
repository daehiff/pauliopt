import unittest

from pauliopt.pauli.anneal import anneal
from pauliopt.topologies import Topology
from tests.utils import generate_all_combination_pauli_polynomial, pauli_poly_to_tket, \
    verify_equality


class TestPauliAnnealing(unittest.TestCase):
    def test_simulated_annealing_pauli(self):
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
                our_synth = anneal(pp, topology)

                self.assertTrue(verify_equality(tket_pp, our_synth),
                                "The annealing version returned a wrong circuit")
