import itertools
from enum import Enum

import networkx as nx

import matplotlib.pyplot as plt
import networkx as nx
import stim
from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit.circuit import Gate, CircuitInstruction
from qiskit.providers.fake_provider import FakeVigo, FakeMumbai

from pauliopt.pauli.anneal import anneal
from pauliopt.pauli.clifford_tableau import CliffordTableau
from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import *
from pauliopt.pauli.utils import Pauli
from pauliopt.utils import pi


def reconstruct_tableau(tableau: stim.Tableau):
    xsx, xsz, zsx, zsz, x_signs, z_signs = tableau.to_numpy()
    x_row = np.concatenate([xsx, xsz], axis=1)
    z_row = np.concatenate([zsx, zsz], axis=1)
    return np.concatenate([x_row, z_row], axis=0).astype(np.int64)


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
        allowed_angels = [2 * pi, pi, pi / 2, pi / 4, pi / 8, pi / 16, pi / 32, pi / 64,
                          pi / 128, pi / 256]

    pp = PauliPolynomial(num_qubits)
    for _ in range(num_gadgets):
        pp >>= create_random_phase_gadget(num_qubits, min_legs, max_legs, allowed_angels)

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
    return np.allclose(Operator.from_circuit(qc_in).data,
                       Operator.from_circuit(qc_out).data)


def verify_equality(qc_in, qc_out):
    """
    Verify the equality up to a global phase
    :param qc_in:
    :param qc_out:
    :return:
    """
    try:
        from qiskit.quantum_info import Statevector
    except:
        raise Exception("Please install qiskit to compare to quantum circuits")
    return Statevector.from_instruction(qc_in) \
        .equiv(Statevector.from_instruction(qc_out))


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


def are_non_zeros_clifford(matrix: np.array):
    matrix_ = np.round(matrix, decimals=np.finfo(matrix.dtype).precision - 1)
    for non_zero in matrix_[matrix_.nonzero()]:
        if non_zero not in [1.0, -1.0, 1.j, -1.j]:
            return False
    return True


def generate_pauli(j: int, n: int, p_type="x"):
    assert p_type == "x" or p_type == "z"
    pauli_x = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    pauli_z = np.asarray([[1.0, 0.0], [0.0, -1.0]])
    if j == 0:
        pauli = pauli_x if p_type == "x" else pauli_z
    else:
        pauli = np.identity(2)

    for i in range(1, n):
        if i == j:
            pauli = np.kron(pauli, pauli_x if p_type == "x" else pauli_z)
        else:
            pauli = np.kron(pauli, np.identity(2))
    return pauli


def is_clifford(gate: Gate):
    gate_unitary = gate.to_matrix()
    for j in range(gate.num_qubits):
        pauli_x_j = generate_pauli(j, gate.num_qubits, p_type="x")
        pauli_z_j = generate_pauli(j, gate.num_qubits, p_type="z")
        if not are_non_zeros_clifford(gate_unitary @ pauli_x_j @ gate_unitary.conj().T) or \
                not are_non_zeros_clifford(
                    gate_unitary @ pauli_z_j @ gate_unitary.conj().T):
            return False

    return True


def count_single_qubit_cliffords(qc: QuantumCircuit):
    count = 0
    for instr, _, _ in qc.data:
        assert isinstance(instr, Gate)
        if is_clifford(instr) and instr.num_qubits < 2:
            count += 1
    return count


def count_single_qubit_non_cliffords(qc: QuantumCircuit):
    count = 0
    for instr, _, _ in qc.data:
        assert isinstance(instr, Gate)
        if not is_clifford(instr) and instr.num_qubits < 2:
            count += 1
    return count


def two_qubit_count(count_ops):
    two_qubit_ops = ["cy", "cx", "cz", "swap", "cz"]
    count = 0
    for op in count_ops:
        if op in two_qubit_ops:
            count += count_ops[op]
    return count


def analyse_ops(qc: QuantumCircuit):
    two_qubit = two_qubit_count(qc.count_ops())
    single_qubit = count_single_qubit_non_cliffords(qc)
    clifford = count_single_qubit_cliffords(qc)
    return {
        "two_qubit": two_qubit,
        "single_qubit": single_qubit,
        "single_qubit_clifford": clifford
    }


def main(n_qubits=5):
    topo = Topology.line(n_qubits)
    pp = generate_random_pauli_polynomial(n_qubits, 200)

    qc_base = pp.to_qiskit(topo)

    qc_opt = anneal(pp, topo, nr_iterations=1500)
    print("Base: ", analyse_ops(qc_base))
    print("Opt : ", analyse_ops(qc_opt))
    print(verify_equality(qc_base, qc_opt))


def n_times_kron(p_list):
    a = p_list[0]
    for pauli in p_list[1:]:
        a = np.kron(a, pauli)
    return a


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
        elif gate_t == "CY":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cy(control, target)
        elif gate_t == "CZ":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cz(control, target)
    return qc


class GateType(Enum):
    H = 0
    S = 1
    CX = 2
    Rx = 3


class Gate:
    def __init__(self, type, qubits):
        self.type = type
        self.qubits = qubits

    @staticmethod
    def from_op(op: CircuitInstruction):
        if op.operation.name == "h":
            return Gate(GateType.H, [op.qubits[0].index])
        elif op.operation.name == "s":
            return Gate(GateType.S, [op.qubits[0].index])
        elif op.operation.name == "cx":
            return Gate(GateType.CX, [op.qubits[0].index, op.qubits[1].index])
        elif op.operation.name == "rz":
            return Gate(GateType.Rx, [op.qubits[0].index])


def build_model_circuit(n):
    """Create quantum fourier transform circuit on quantum register qreg."""
    qreg = QuantumRegister(n)
    circuit = QuantumCircuit(qreg, name="qft")

    for i in range(n):
        for j in range(i):
            # Using negative exponents so we safely underflow to 0 rather than
            # raise `OverflowError`.
            circuit.cp(math.pi * (2.0 ** (j - i)), qreg[i], qreg[j])
        circuit.h(qreg[i])

    return circuit


def pauli_string_to_matrix(pauli_string):
    if pauli_string == "I":
        return np.eye(2)
    elif pauli_string == "X":
        return np.array([[0, 1], [1, 0]])
    elif pauli_string == "Y":
        return np.array([[0, -1j], [1j, 0]])
    elif pauli_string == "Z":
        return np.array([[1, 0], [0, -1]])
    else:
        raise Exception("Invalid pauli string")


def get_resulting_pauli(res_matrix, pauli2):
    for sign in [-1, 1, 1.j, -1.j]:
        for pauli in ["X", "Y", "Z", "I"]:
            if np.allclose(res_matrix, sign * pauli_string_to_matrix(pauli)):
                return sign, pauli


def main():
    # backend = FakeMumbai()

    PAULI_DICT = {}
    for combinations in itertools.product(["X", "Y", "Z", "I"], repeat=2):
        res_matrix = pauli_string_to_matrix(combinations[0]) @ \
                     pauli_string_to_matrix(combinations[1])
        sign, pauli = get_resulting_pauli(res_matrix, combinations)
        PAULI_DICT[combinations] = (sign, pauli)

    print(PAULI_DICT)

    # circ = QuantumCircuit.from_qasm_file("test.qasm")
    #
    # topo = Topology.line(circ.num_qubits)
    #
    # # circ = QuantumCircuit.from_qasm_file("test.qasm")
    #
    # # nx.draw(topo.to_nx, with_labels=True)
    # # plt.show()
    #
    # circ_out = transpile(circ,
    #                      basis_gates=["h", "s", "cx"],
    #                      approximation_degree=1.0,
    #                      coupling_map=[[i, j] for (i, j) in topo.to_nx.edges()],
    #                      optimization_level=0)
    # print("Qiskit:", circ_out.count_ops())
    #
    # circ = transpile(circ, basis_gates=["h", "s", "cx"], optimization_level=0)
    # ct = CliffordTableau.from_circuit(circ)
    # circ_out = ct.to_cifford_circuit_arch_aware(topo)
    # print("Clifford:", circ_out.count_ops())
    # print(verify_equality(circ, circ_out))
    # print("Original: ", circ.count_ops())
    # print("Tableau:  ", circ_out.count_ops())
    # print("Transpile: ", circ_.count_ops())
    # assert verify_equality(circ, circ_out)


if __name__ == '__main__':
    main()
