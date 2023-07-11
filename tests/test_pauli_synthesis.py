import unittest
import numpy as np
from qiskit import QuantumCircuit

from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import PauliPolynomial, simplify_pauli_polynomial
from pauliopt.pauli.synth.anneal import anneal
from pauliopt.pauli.synth.arch_aware_uccds import uccds_synthesis
from pauliopt.pauli.synth.divide_conquer import synth_divide_and_conquer
from pauliopt.pauli.synth.tableau_synth import pauli_polynomial_steiner_gray_synth_nc
from pauliopt.pauli.utils import X, Y, Z, I, apply_permutation
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


def verify_equality(qc_in, qc_out):
    try:
        from qiskit.quantum_info import Statevector
    except:
        raise Exception("Please install qiskit to compare to quantum circuits")
    return Statevector.from_instruction(qc_in) \
        .equiv(Statevector.from_instruction(qc_out))


def check_matching_architecture(qc: QuantumCircuit, G):
    for gate in qc:
        if gate.operation.num_qubits == 2:
            ctrl, target = gate.qubits
            ctrl, target = ctrl._index, target._index  # TODO refactor this to a non deprecated way
            if not G.has_edge(ctrl, target):
                return False
    return True


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
            circ_out, gadget_perm, perm = uccds_synthesis(pp.copy(), topo)
            pp_ = PauliPolynomial(pp.num_qubits)
            pp_.pauli_gadgets = [pp[i] for i in gadget_perm]
            self.assertTrue(
                verify_equality(circ_out, pp_.to_qiskit(topology=topo)),
                "Circuits did not match")
            circ_out = apply_permutation(circ_out, perm)
            self.assertTrue(check_matching_architecture(circ_out, topo.to_nx),
                            "architecture did not match")

    def test_divide_and_conquer(self):
        for num_gadgets in [10, 30]:
            for topo in [Topology.line(4),
                         Topology.cycle(4),
                         Topology.complete(4),
                         Topology.grid(2, 3)]:
                print(topo._named)
                pp = generate_random_pauli_polynomial(topo.num_qubits, num_gadgets)
                circ_out, _, perm = \
                    synth_divide_and_conquer(pp.copy(), topo)
                self.assertTrue(check_matching_architecture(circ_out, topo.to_nx),
                                "architecture did not match")
                circ_out = apply_permutation(circ_out, perm)
                self.assertTrue(
                    verify_equality(circ_out, pp.to_qiskit(topology=topo)),
                    "Circuits did not match")

    def test_steiner_divide_and_conquer(self):
        for num_gadgets in [100, 200]:
            for topo in [Topology.line(4),
                         Topology.line(6),
                         Topology.cycle(4),
                         Topology.grid(2, 4)]:
                print(topo._named)
                pp = generate_random_pauli_polynomial(topo.num_qubits, num_gadgets)
                circ_out, gadget_perm, perm = \
                    pauli_polynomial_steiner_gray_synth_nc(pp.copy(), topo)
                pp_ = PauliPolynomial(pp.num_qubits)
                pp_.pauli_gadgets = [pp[i] for i in gadget_perm]
                self.assertTrue(
                    verify_equality(circ_out, pp_.to_qiskit(topology=topo)),
                    "Circuits did not match")
                circ_out = apply_permutation(circ_out, perm)
                self.assertTrue(check_matching_architecture(circ_out, topo.to_nx),
                                "architecture did not match")

    def test_pauli_annealing(self):
        for num_gadgets in [100, 200]:
            for topo in [Topology.line(4),
                         Topology.line(6),
                         Topology.cycle(4),
                         Topology.grid(2, 4)]:
                print(topo._named)
                pp = generate_random_pauli_polynomial(topo.num_qubits, num_gadgets)
                circ_out, gadget_perm, perm = anneal(pp.copy(), topo)
                pp_ = PauliPolynomial(pp.num_qubits)
                pp_.pauli_gadgets = [pp[i] for i in gadget_perm]
                self.assertTrue(
                    verify_equality(circ_out, pp_.to_qiskit(topology=topo)),
                    "Circuits did not match")
                circ_out = apply_permutation(circ_out, perm)
                self.assertTrue(check_matching_architecture(circ_out, topo.to_nx),
                                "architecture did not match")
