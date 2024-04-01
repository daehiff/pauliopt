import itertools
import unittest

import networkx as nx
import numpy as np
import pytket
from pytket._tket.circuit import PauliExpBox
from pytket._tket.transform import Transform
from pytket.extensions.qiskit.qiskit_convert import tk_to_qiskit, qiskit_to_tk
from pytket._tket.pauli import Pauli
from qiskit import QuantumCircuit

from pauliopt.pauli.pauli_circuit import CliffordType, generate_random_clifford
from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import PauliPolynomial, simplify_pauli_polynomial
from pauliopt.pauli.utils import X, Y, Z, I

from pauliopt.topologies import Topology
from pauliopt.utils import pi

PAULI_TO_TKET = {X: Pauli.X, Y: Pauli.Y, Z: Pauli.Z, I: Pauli.I}


def tket_to_qiskit(circuit: pytket.Circuit) -> QuantumCircuit:
    return tk_to_qiskit(circuit)


def pauli_poly_to_tket(pp: PauliPolynomial):
    circuit = pytket.Circuit(pp.num_qubits)
    for gadget in pp.pauli_gadgets:
        circuit.add_pauliexpbox(
            PauliExpBox(
                [PAULI_TO_TKET[p] for p in gadget.paulis],
                gadget.angle.to_qiskit / np.pi,
            ),
            list(range(pp.num_qubits)),
        )
    Transform.DecomposeBoxes().apply(circuit)
    return tket_to_qiskit(circuit)


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
    return Operator.from_circuit(qc_in).equiv(Operator.from_circuit(qc_out))


def create_random_phase_gadget(num_qubits, min_legs, max_legs, allowed_angels):
    angle = np.random.choice(allowed_angels)
    nr_legs = np.random.randint(min_legs, max_legs)
    legs = np.random.choice([i for i in range(num_qubits)], size=nr_legs, replace=False)
    phase_gadget = [I for _ in range(num_qubits)]
    for leg in legs:
        phase_gadget[leg] = np.random.choice([X, Y, Z])
    return PPhase(angle) @ phase_gadget


def generate_random_pauli_polynomial(
    num_qubits: int, num_gadgets: int, min_legs=None, max_legs=None, allowed_angels=None
):
    if min_legs is None:
        min_legs = 1
    if max_legs is None:
        max_legs = num_qubits
    if allowed_angels is None:
        allowed_angels = [
            2 * pi,
            pi,
            pi / 2,
            pi / 4,
            pi / 8,
            pi / 16,
            pi / 32,
            pi / 64,
            pi / 128,
            pi / 256,
        ]

    pp = PauliPolynomial(num_qubits)
    for _ in range(num_gadgets):
        pp >>= create_random_phase_gadget(
            num_qubits, min_legs, max_legs, allowed_angels
        )

    return pp


def generate_all_combination_pauli_polynomial(n_qubits=2):
    allowed_angels = [2 * pi, pi, pi / 2, pi / 4, pi / 8]
    pp = PauliPolynomial(n_qubits)
    for comb in itertools.product([X, Y, Z, I], repeat=n_qubits):
        pp >>= PPhase(np.random.choice(allowed_angels)) @ list(comb)
    return pp


def check_matching_architecture(qc: QuantumCircuit, G: nx.Graph):
    for gate in qc:
        if gate.operation.num_qubits == 2:
            ctrl, target = gate.qubits
            ctrl, target = (
                ctrl._index,
                target._index,
            )  # TODO refactor this to a non deprecated way
            if not G.has_edge(ctrl, target):
                return False
    return True


def get_two_qubit_count(circ: QuantumCircuit):
    ops = circ.count_ops()
    two_qubit_count = 0
    two_qubit_ops = ["cx", "cy", "cz"]
    for op_key in two_qubit_ops:
        if op_key in ops.keys():
            two_qubit_count += ops[op_key]

    return two_qubit_count


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
                self.assertTrue(
                    verify_equality(tket_pp, our_synth),
                    "The resulting Quantum Circuits were not equivalent",
                )
                self.assertTrue(
                    check_matching_architecture(our_synth, topology.to_nx),
                    "The Pauli Polynomial did not match the architecture",
                )
                self.assertEqual(
                    our_synth.count_ops()["cx"],
                    pp.two_qubit_count(topology),
                    "Two qubit count needs to be equivalent to to two qubit count of the circuit",
                )

    def test_gate_propagation(self):
        """
        Checks if the clifford Propagation rules are sound for 2, 3, 4 qubits
        """
        for num_qubits in [2, 3, 4]:
            pp = generate_all_combination_pauli_polynomial(n_qubits=num_qubits)
            inital_qc = pp.to_qiskit()
            for gate_class in [
                CliffordType.CX,
                CliffordType.CY,
                CliffordType.CZ,
                CliffordType.H,
                CliffordType.S,
                CliffordType.V,
                CliffordType.Sdg,
                CliffordType.Vdg,
            ]:
                gate = generate_random_clifford(gate_class, num_qubits)
                print(gate_class, gate)
                pp_ = pp.copy().propagate(gate)
                qc = QuantumCircuit(num_qubits)
                gate_qiskit, qubits = gate.to_qiskit()
                qc_gate = QuantumCircuit(pp.num_qubits)
                qc_gate.append(gate_qiskit, qubits)
                qc.compose(qc_gate.inverse(), inplace=True)
                qc.compose(pp_.to_qiskit(), inplace=True)
                qc.compose(qc_gate, inplace=True)
                self.assertTrue(
                    verify_equality(inital_qc, qc),
                    "The resulting Quantum Circuits were not equivalent",
                )

    def test_pauli_polynomial_simplification(self):
        """
        Checks the simplification process of a Pauli Polynomial.
        """
        for num_qubits in [2, 3, 4]:
            for nr_gadets in [100, 200, 300]:
                pp = generate_random_pauli_polynomial(num_qubits, nr_gadets)

                pp_simplified = simplify_pauli_polynomial(pp.copy())

                qc_pp = pp.to_qiskit()
                qc_pp_simplified = pp_simplified.to_qiskit()
                self.assertTrue(
                    verify_equality(qc_pp, qc_pp_simplified),
                    "The resulting Quantum Circuits were not equivalent",
                )

    def test_pauli_construction(self):
        num_qubits = 4
        topology = Topology.complete(num_qubits)
        pp = generate_random_pauli_polynomial(num_qubits, 5)
        tket_pp = pauli_poly_to_tket(pp)
        our_synth = pp.to_qiskit(topology)

        print(tket_pp)
        print(our_synth)

        self.assertTrue(
            verify_equality(tket_pp, our_synth),
            "The resulting Quantum Circuits were not equivalent",
        )
