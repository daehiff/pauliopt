import unittest

from pytket._tket.architecture import Architecture
from pytket._tket.passes import PauliSimp, PlacementPass, RoutingPass, \
    FullPeepholeOptimise, DecomposeBoxes
from pytket._tket.placement import GraphPlacement
from pytket.extensions.qiskit import tk_to_qiskit

from pauliopt.pauli.tket import synth_tket
from pauliopt.topologies import Topology
from tests.utils import generate_all_combination_pauli_polynomial, pauli_poly_to_tket, \
    verify_equality, check_matching_architecture, generate_random_phase_poly


class TestTketSynthesis(unittest.TestCase):
    def test_tket_synth(self):
        for num_qubits in [2, 3, 4]:
            for topo_creation in [Topology.line, Topology.complete]:
                pp = generate_random_phase_poly(n_qubits=num_qubits, n_gadgets=20)
                topology = topo_creation(pp.num_qubits)

                tket_arch = Architecture([e for e in topology.to_nx.edges()])
                custom_method = [
                    DecomposeBoxes(),
                    FullPeepholeOptimise(allow_swaps=False),
                    RoutingPass(tket_arch),
                ]
                for method in ["PauliSimp", custom_method]:
                    tket_pp = pauli_poly_to_tket(pp)
                    our_synth = synth_tket(pp, topology, method=method)

                    self.assertTrue(verify_equality(tket_pp, our_synth),
                                    "The annealing version returned a wrong circuit")

                    unit = synth_tket(pp, topology, method=method, return_circuit=False)

                    our_synth = tk_to_qiskit(unit.circuit)
                    self.assertTrue(
                        check_matching_architecture(our_synth, topology.to_nx),
                        "The annealing version returned a wrong circuit")
