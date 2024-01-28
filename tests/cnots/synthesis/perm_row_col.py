import unittest

import numpy as np
from parameterized import parameterized

from pauliopt.circuits import Circuit
from pauliopt.cnots.parity_map import ParityMap
from pauliopt.topologies import Topology
from tests.utils import random_hscx_circuit, random_clifford_circuit, verify_equality
from pauliopt.cnots.synthesis.perm_row_col import _perm_row_col as perm_row_col


class TestTableauSynthesis(unittest.TestCase):
    @parameterized.expand(
        [
            ("line_5", 5, 5, Topology.line(5), True),
            ("line_6", 6, 1000, Topology.line(6), True),
            ("line_8", 8, 1000, Topology.line(8), True),
            ("grid_4", 4, 1000, Topology.grid(2, 2), True),
            ("grid_8", 8, 1000, Topology.grid(2, 4), True),
            ("line_5", 5, 1000, Topology.line(5), False),
            ("line_8", 8, 1000, Topology.line(8), False),
            ("grid_4", 4, 1000, Topology.grid(2, 2), False),
            ("grid_8", 8, 1000, Topology.grid(2, 4), False),
        ]
    )
    def test_clifford_synthesis(self, _, n_qubits, n_gates, topo, reallocate):
        print("===")
        circuit = random_clifford_circuit(
            nr_qubits=n_qubits, nr_gates=n_gates, gate_choice=["CX"]
        )

        circuit = Circuit.from_qiskit(circuit)

        parity_map = ParityMap.from_circuit_append(circuit)

        circ_1 = perm_row_col(parity_map, topo, reallocate=reallocate)

        circuit = circuit.to_qiskit()
        circ_1 = circ_1.to_qiskit()

        self.assertTrue(
            verify_equality(circuit, circ_1),
            "The Synthesized circuit does not equal to original",
        )
