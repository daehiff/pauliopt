import warnings

from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import random_clifford, Clifford

warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np

import time

import qiskit.quantum_info as qi

from pauliopt.pauli.clifford_tableau import CliffordTableau
from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import *
from pauliopt.utils import pi, AngleVar


def generate_random_z_polynomial(
    num_qubits: int, num_gadgets: int, min_legs=None, max_legs=None, allowed_angels=None
):
    if min_legs is None:
        min_legs = 2
    if max_legs is None:
        max_legs = num_qubits
    if allowed_angels is None:
        allowed_angels = [2 * pi, pi, pi / 2, pi / 4, pi / 8, pi / 16, pi / 32, pi / 64]
    allowed_legs = [Z]
    pp = PauliPolynomial(num_qubits)
    for _ in range(num_gadgets):
        pp >>= create_random_phase_gadget(
            num_qubits, min_legs, max_legs, allowed_angels, allowed_legs=allowed_legs
        )
    return pp


def create_random_phase_gadget(
    num_qubits, min_legs, max_legs, allowed_angels, allowed_legs=None
):
    if allowed_legs is None:
        allowed_legs = [X, Y, Z]
    angle = np.random.choice(allowed_angels)
    nr_legs = np.random.randint(min_legs, max_legs)
    legs = np.random.choice([i for i in range(num_qubits)], size=nr_legs, replace=False)
    phase_gadget = [I for _ in range(num_qubits)]
    for leg in legs:
        phase_gadget[leg] = np.random.choice(allowed_legs)
    return PPhase(angle) @ phase_gadget


def generate_random_pauli_polynomial(
    num_qubits: int, num_gadgets: int, min_legs=None, max_legs=None, allowed_angels=None
):
    if min_legs is None:
        min_legs = 1
    if max_legs is None:
        max_legs = num_qubits
    if allowed_angels is None:
        allowed_angels = [pi, pi / 2, pi / 4, pi / 8, pi / 16]

    pp = PauliPolynomial(num_qubits)
    for _ in range(num_gadgets):
        pp >>= create_random_phase_gadget(
            num_qubits, min_legs, max_legs, allowed_angels
        )

    return pp


def verify_equality_unitary(qc_in, qc_out):
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
    return np.allclose(
        Operator.from_circuit(qc_in).data, Operator.from_circuit(qc_out).data
    )


def verify_equality(qc_in, qc_out):
    """
    Verify the equality up to a global phase
    :param qc_in:
    :param qc_out:
    :return:
    """
    try:
        from qiskit.quantum_info import Statevector, Operator
    except:
        raise Exception("Please install qiskit to compare to quantum circuits")
    return Statevector.from_instruction(qc_in).equiv(
        Statevector.from_instruction(qc_out)
    )


def random_hscx_circuit(nr_gates=20, nr_qubits=4):
    gate_choice = ["CX", "H", "S"]
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
        elif gate_t == "CY":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cy(control, target)
        elif gate_t == "CZ":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cz(control, target)
    return qc


def permute_pp(pp: PauliPolynomial, permutation: list):
    swapped = [False] * pp.num_qubits

    for idx, i in enumerate(permutation):
        if i != idx and not swapped[i] and not swapped[idx]:
            swapped[i] = True
            swapped[idx] = True
            pp.swap_rows(idx, i)
    return pp


def clifford_tableau_fun():
    matrix = np.array(
        [
            [0, 0, 1, 0, 0, 1],
            [1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0],
        ]
    )
    signs = np.array([0, 0, 0, 0, 0, 0])

    ct = CliffordTableau(tableau=matrix, signs=signs)
    ct.append_s(2)
    ct.append_cnot(2, 0)
    ct.append_cnot(0, 2)
    ct.append_cnot(2, 0)
    ct.append_cnot(1, 2)
    ct.print_zx()


def pp_decomposition():
    pp = PauliPolynomial(2)

    pp >>= PPhase(AngleVar("\\alpha_0")) @ [Y, Z]
    pp >>= PPhase(AngleVar("\\alpha_1")) @ [Z, Z]

    print(pp.to_latex())


def test_clifford_1001():
    start = time.time()
    circ = random_hscx_circuit(10, 5)
    topo = Topology.line(circ.num_qubits)
    coupling = [list(x) for x in topo.couplings]

    circ_ = transpile(
        circ,
        initial_layout=[x for x in range(circ.num_qubits)],
        basis_gates=["cx", "h", "s"],
        coupling_map=coupling,
    )
    print("Original: ", circ.count_ops())

    ct = CliffordTableau.from_circuit(circ)
    ct.print_zx()

    circ_out_opt = ct.optimal_to_circuit(topo)

    circ_out_ours, _ = ct.to_clifford_circuit_arch_aware_qiskit(
        topo, include_swaps=False
    )
    tableau = Clifford.from_circuit(circ)
    circ_out_bravi = tableau.to_circuit()
    circ_out_bravi = transpile(
        circ_out_bravi,
        initial_layout=[x for x in range(circ.num_qubits)],
        coupling_map=coupling,
        basis_gates=["cx", "h", "s"],
    )

    print(verify_equality(circ_out_opt, circ))

    print("Optimal:   ", circ_out_opt.count_ops()["cx"])
    print("Bravi:     ", circ_out_bravi.count_ops()["cx"])
    print("Ours:      ", circ_out_ours.count_ops()["cx"])
    print("Original:  ", circ_.count_ops()["cx"])
    print("dt:        ", time.time() - start)


def test_tableau_1002(n_qubits=6):
    topo = Topology.line(n_qubits)
    clifford = random_clifford(n_qubits)

    circ_bravi = clifford.to_circuit()
    circ_bravi = transpile(circ_bravi, basis_gates=["h", "s", "cx"])

    circ_bravi.qasm(filename="test.qasm")

    circ_bravi = QuantumCircuit.from_qasm_file("test.qasm")
    # exit(0)
    ct = CliffordTableau.from_circuit(circ_bravi)
    circ_out, perm = ct.to_cifford_circuit_perm_row_col(topo, include_swaps=True)
    circ_out = circ_out.to_qiskit()
    assert (
        qi.Operator.from_circuit(circ_out).equiv(qi.Operator.from_circuit(circ_bravi))
    )


if __name__ == "__main__":
    for _ in range(1000):
        test_tableau_1002(6)
