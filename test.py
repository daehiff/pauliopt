from enum import Enum

import galois
import pytket
import stim
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from qiskit import QuantumCircuit, QuantumRegister, transpile, IBMQ
from qiskit.circuit import Gate, CircuitInstruction
from qiskit.providers.fake_provider import FakeMumbai

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


def undo_permutation(qc: QuantumCircuit, perm):
    circ_out = qiskit_to_tk(qc)
    inv_map = {circ_out.qubits[v]: circ_out.qubits[k] for v, k in enumerate(perm)}
    circ_out_ = pytket.Circuit(circ_out.n_qubits)
    for cmd in circ_out:
        if isinstance(cmd, pytket.circuit.Command):
            remaped_qubits = list(map(lambda node: inv_map[node], cmd.qubits))
            circ_out_.add_gate(cmd.op, remaped_qubits)
    circ_out = tk_to_qiskit(circ_out_)
    return circ_out


def main():
    # login to IBMQ
    #IBMQ.load_account()


    circ = QuantumCircuit.from_qasm_file("test6.qasm")
    #circ = random_hscx_circuit(10, 5)
    #circ.qasm(filename="test6.qasm")
    topo = Topology.line(circ.num_qubits)
    remaining = CliffordTableau.from_circuit(circ)

    circ_out, perm = remaining.to_cifford_circuit_arch_aware(topo)

    print(circ_out)
    print(perm)
    circ_out = undo_permutation(circ_out, perm)
    print(verify_equality(circ, circ_out))
    assert verify_equality(circ, circ_out)
    circ = transpile(circ, basis_gates=["h", "s", "cx", "rz"],
                     coupling_map=[[e1, e2] for e1, e2 in topo.to_nx.edges()])
    print(circ)
    print(circ.count_ops())
    print(circ_out.count_ops())


if __name__ == '__main__':
    main()
