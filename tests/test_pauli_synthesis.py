import unittest

import numpy as np

from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.synthesis import PauliSynthesizer, SynthMethod
from pauliopt.pauli.utils import X, Y, Z, I
from pauliopt.topologies import Topology
from pauliopt.utils import pi


def create_random_phase_gadget(num_qubits, min_legs, max_legs, allowed_angels,
                               allowed_legs=None):
    if allowed_legs is None:
        allowed_legs = [X, Y, Z]
    angle = np.random.choice(allowed_angels)
    nr_legs = np.random.randint(min_legs, max_legs)
    legs = np.random.choice([i for i in range(num_qubits)], size=nr_legs, replace=False)
    phase_gadget = [I for _ in range(num_qubits)]
    for leg in legs:
        phase_gadget[leg] = np.random.choice(allowed_legs)
    return PPhase(angle) @ phase_gadget


def generate_random_pauli_polynomial(num_qubits: int, num_gadgets: int, min_legs=None,
                                     max_legs=None, allowed_angels=None):
    if min_legs is None:
        min_legs = 1
    if max_legs is None:
        max_legs = num_qubits
    if allowed_angels is None:
        allowed_angels = [pi, pi / 2, pi / 4, pi / 8, pi / 16]

    pp = PauliPolynomial(num_qubits)
    for _ in range(num_gadgets):
        pp >>= create_random_phase_gadget(num_qubits, min_legs, max_legs, allowed_angels)

    return pp


class TestPauliSynthesis(unittest.TestCase):
    def test_uccds(self):
        for num_gadgets in [100, 200]:
            for topo in [Topology.line(4),
                         Topology.line(8),
                         Topology.cycle(4),
                         Topology.cycle(8),
                         Topology.grid(2, 3)]:
                print(topo._named)
                pp = generate_random_pauli_polynomial(topo.num_qubits, num_gadgets)
                synthesizer = PauliSynthesizer(pp, SynthMethod.UCCDS, topo)
                synthesizer.synthesize()
                self.assertTrue(synthesizer.check_circuit_equivalence(),
                                "Circuits did not match")
                self.assertTrue(synthesizer.check_connectivity_predicate(),
                                "Connectivity predicate not satisfied")

    def test_divide_and_conquer(self):
        for num_gadgets in [10, 30]:
            for topo in [Topology.line(4),
                         Topology.cycle(4),
                         Topology.complete(4),
                         Topology.grid(2, 3)]:
                print(topo._named)
                pp = generate_random_pauli_polynomial(topo.num_qubits, num_gadgets)
                synthesizer = PauliSynthesizer(pp, SynthMethod.DIVIDE_AND_CONQUER, topo)
                synthesizer.synthesize()
                self.assertTrue(synthesizer.check_circuit_equivalence(),
                                "Circuits did not match")
                self.assertTrue(synthesizer.check_connectivity_predicate(),
                                "Connectivity predicate not satisfied")

    def test_steiner_gray_nc(self):
        for num_gadgets in [100, 200]:
            for topo in [Topology.line(4),
                         Topology.line(6),
                         Topology.cycle(4),
                         Topology.grid(2, 4)]:
                print(topo._named)
                pp = generate_random_pauli_polynomial(topo.num_qubits, num_gadgets)
                synthesizer = PauliSynthesizer(pp, SynthMethod.STEINER_GRAY_NC, topo)
                synthesizer.synthesize()
                self.assertTrue(synthesizer.check_circuit_equivalence(),
                                "Circuits did not match")
                self.assertTrue(synthesizer.check_connectivity_predicate(),
                                "Connectivity predicate not satisfied")

    def test_pauli_annealing(self):
        for num_gadgets in [100, 200]:
            for topo in [Topology.line(4),
                         Topology.line(6),
                         Topology.cycle(4),
                         Topology.grid(2, 4)]:
                print(topo._named)
                pp = generate_random_pauli_polynomial(topo.num_qubits, num_gadgets)
                synthesizer = PauliSynthesizer(pp, SynthMethod.ANNEAL, topo)
                synthesizer.synthesize()
                self.assertTrue(synthesizer.check_circuit_equivalence(),
                                "Circuits did not match")
                self.assertTrue(synthesizer.check_connectivity_predicate(),
                                "Connectivity predicate not satisfied")
