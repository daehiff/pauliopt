import unittest
import numpy as np

from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import PauliPolynomial, simplify_pauli_polynomial
from pauliopt.pauli.utils import Pauli


def create_random_phase_gadget(num_qubits, min_legs, max_legs, allowed_angels):
    angle = np.random.choice(allowed_angels)
    nr_legs = np.random.randint(min_legs, max_legs)
    legs = np.random.choice([i for i in range(num_qubits)], size=nr_legs, replace=False)
    phase_gadget = [Pauli.I for _ in range(num_qubits)]
    for leg in legs:
        phase_gadget[leg] = np.random.choice([Pauli.X, Pauli.Y, Pauli.Z])
    return PPhase(angle) @ phase_gadget


def generate_random_pauli_polynomial(num_qubits: int, num_gadgets: int, min_legs=None,
                                     max_legs=None, allowed_angels=None):
    if min_legs is None:
        min_legs = 1
    if max_legs is None:
        max_legs = num_qubits
    if allowed_angels is None:
        allowed_angels = [2 * np.pi, np.pi, 0.5 * np.pi, 0.25 * np.pi, 0.125 * np.pi]

    pp = PauliPolynomial(num_qubits)
    for _ in range(num_gadgets):
        pp >>= create_random_phase_gadget(num_qubits, min_legs, max_legs, allowed_angels)

    return pp


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


class TestPauliConversion(unittest.TestCase):
    def test_simplification_process(self):
        for num_qubits in [4, 6]:
            for phase_gadget in [100, 200, 300, 400]:
                pp = generate_random_pauli_polynomial(num_qubits, phase_gadget)

                pp_ = simplify_pauli_polynomial(pp)

                self.assertTrue(verify_equality(pp_.to_qiskit(), pp.to_qiskit()),
                                "Resulting circuits where not equal")
