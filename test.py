import warnings
from enum import Enum

import galois
import numpy as np
import pytket
import stim
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from qiskit import QuantumCircuit, QuantumRegister, transpile, IBMQ, execute, Aer
from qiskit.circuit import Gate, CircuitInstruction
from qiskit.providers.fake_provider import FakeMumbai, FakeVigo
from qiskit.quantum_info import Clifford, StabilizerState, Pauli

from pauliopt.pauli.anneal import anneal, global_leg_removal
from pauliopt.pauli.clifford_gates import CliffordType
from pauliopt.pauli.clifford_tableau import CliffordTableau, reconstruct_tableau_signs
from pauliopt.pauli.divide_conquer import synth_divide_and_conquer
from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import *
from pauliopt.pauli.utils import apply_permutation
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
    phase_gadget = [I for _ in range(num_qubits)]
    for leg in legs:
        phase_gadget[leg] = np.random.choice([X, Y, Z])
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


def n_times_kron(p_list):
    a = p_list[0]
    for pauli in p_list[1:]:
        a = np.kron(a, pauli)
    return a


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


def append_circuit_to_tableau(circ: QuantumCircuit, tableau: stim.Tableau):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    cnot = stim.Tableau.from_named_gate("CX")
    had = stim.Tableau.from_named_gate("H")
    s = stim.Tableau.from_named_gate("S")
    for op in circ:
        if op.operation.name == "h":
            tableau.append(had, [op.qubits[0].index])
        elif op.operation.name == "s":
            tableau.append(s, [op.qubits[0].index])
        elif op.operation.name == "cx":
            tableau.append(cnot, [op.qubits[0].index, op.qubits[1].index])
        else:
            raise Exception("Unknown operation")
    return tableau


def prepend_circuit_to_tableau(circ: QuantumCircuit, tableau: stim.Tableau):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    cnot = stim.Tableau.from_named_gate("CX")
    had = stim.Tableau.from_named_gate("H")
    s = stim.Tableau.from_named_gate("S")
    for op in reversed(circ):
        if op.operation.name == "h":
            tableau.prepend(had, [op.qubits[0].index])
        elif op.operation.name == "s":
            tableau.prepend(s, [op.qubits[0].index])
        elif op.operation.name == "cx":
            tableau.prepend(cnot, [op.qubits[0].index, op.qubits[1].index])
        else:
            raise Exception("Unknown operation")
    return tableau


def parse_stim_to_qiskit(circ: stim.Circuit):
    qc = QuantumCircuit(circ.num_qubits)
    for gate in circ:
        if gate.name == "CX":
            targets = [target.value for target in gate.targets_copy()]
            targets = [(targets[i], targets[i + 1]) for i in range(0, len(targets), 2)]
            for (ctrl, target) in targets:
                qc.cx(ctrl, target)
        elif gate.name == "H":
            targets = [target.value for target in gate.targets_copy()]
            for qubit in targets:
                qc.h(qubit)
        elif gate.name == "S":
            targets = [target.value for target in gate.targets_copy()]
            for qubit in targets:
                qc.s(qubit)
        else:
            raise TypeError(f"Unknown Name: {gate.name}")
    return qc


def test_prepedning():
    circ_prev = random_hscx_circuit(1000, 8)
    circ_prev.s(0)
    circ_prev.s(0)

    # circ_prev.cx(1, 0)
    # circ_prev.h(1)
    # circ_prev.h(1)

    # circ_prev.cx(0, 1)
    # circ_prev.h(1)
    # circ_next = random_hscx_circuit(1000, 4)
    tableau = stim.Tableau(circ_prev.num_qubits)

    # tableau = append_circuit_to_tableau(circ_next, tableau)
    tableau = prepend_circuit_to_tableau(circ_prev, tableau)
    tableau_matrix = reconstruct_tableau(tableau)
    tableau_signs = reconstruct_tableau_signs(tableau)
    tableau = CliffordTableau(tableau=tableau_matrix, signs=tableau_signs)
    circ_stim = tableau.to_clifford_circuit()

    ct = CliffordTableau(circ_prev.num_qubits)
    # ct.append_circuit(circ_next)
    ct.prepend_circuit(circ_prev)
    circ_ct = ct.to_clifford_circuit()
    # #
    # print(ct.tableau)
    # print(tableau_matrix)
    assert np.allclose(ct.tableau, tableau_matrix), "Matrices don't match"

    # print(ct.signs)
    # print(tableau_signs)
    assert np.allclose(ct.signs, tableau_signs), "Signs didn't match"
    circ_out = circ_prev
    # circ_out = circ_prev.compose(circ_next)
    # print(circ_prev)
    # print(circ_next)
    # print(circ_out)

    assert (verify_equality(circ_ct, circ_out))


def test_clifford_synthesis():
    circ = random_hscx_circuit(1000, 8)

    ct = CliffordTableau.from_circuit(circ)
    topo = Topology.line(circ.num_qubits)

    circ_out, perm = ct.to_cifford_circuit_arch_aware(topo)

    assert verify_equality(circ, circ_out)


def check_matching_architecture(qc: QuantumCircuit, G):
    for gate in qc:
        if gate.operation.num_qubits == 2:
            ctrl, target = gate.qubits
            ctrl, target = ctrl._index, target._index  # TODO refactor this to a non deprecated way
            if not G.has_edge(ctrl, target):
                return False
    return True


def random_rotation_circuit(num_gates, num_qubits):
    circ = QuantumCircuit(num_qubits)
    for i in range(num_gates):
        qubit = np.random.randint(0, num_qubits)
        angle = np.random.rand() * 2 * np.pi
        gate = np.random.choice(["rx", "ry", "rz"])
        if gate == "rx":
            circ.rx(angle, qubit)
        elif gate == "rz":
            circ.rz(angle, qubit)
        else:
            circ.ry(angle, qubit)
    return circ


def transform_clifford_basis(measurements, clifford_circuit, shots=100):
    new_measurements = {}
    for measure, m_value in measurements.items():
        p_string = "".join(["X" if m == '1' else "Z" for m in measure])
        pauli = Pauli(p_string)
        state = StabilizerState(pauli).evolve(clifford_circuit)
        c_measurement = state.probabilities_dict()
        # m_value = m_value / len(c_measurement)
        for k, v in c_measurement.items():
            if k in new_measurements:
                new_measurements[k] += v * m_value
            else:
                new_measurements[k] = v * m_value
    sum = 0
    for k, v in new_measurements.items():
        sum += v
    print(sum)
    return new_measurements


def test_circuit_simulation():
    # clifford_circ = QuantumCircuit(2)
    # clifford_circ.cx(0, 1)
    # clifford_circ.cx(1, 0)
    clifford_circ = random_hscx_circuit(4, 2)
    rotation_circ = random_rotation_circuit(10, 2)
    circ = QuantumCircuit(8)
    circ.compose(clifford_circ.inverse(), inplace=True)
    circ.compose(rotation_circ, inplace=True)
    circ.compose(clifford_circ, inplace=True)
    print(circ)

    result = execute(circ, Aer.get_backend("statevector_simulator")).result()
    counts_ = result.get_counts()
    print(counts_)

    result = execute(rotation_circ, Aer.get_backend("statevector_simulator")).result()
    counts = result.get_counts()
    counts = transform_clifford_basis(counts, clifford_circ)
    print(counts)
    assert len(counts) == len(counts_)
    for k in counts.keys():
        assert k in counts_.keys()
        assert np.allclose(counts_[k], counts[k])


def main():
    backend = FakeVigo()
    pp = generate_random_pauli_polynomial(8, 10)
    print(pp.num_legs())
    pp = simplify_pauli_polynomial(pp)
    print(pp.num_legs())
    pp = global_leg_removal(pp, gate_set=[CliffordType.CX, CliffordType.CY,
                                          CliffordType.CZ, CliffordType.CXH])
    print(pp.num_legs())


def test():
    pp = generate_random_pauli_polynomial(4, 200)
    topo = Topology.complete(pp.num_qubits)

    circ_in = pp.to_qiskit(topo)
    pp = simplify_pauli_polynomial(pp)

    circ_out = synth_divide_and_conquer(pp.copy(), topo, add_sort=False)
    print("Ours:      ", circ_out.count_ops())
    # circ_out = synth_divide_and_conquer(pp.copy(), topo, add_sort=True)
    # print("Ours sort: ", circ_out.count_ops())
    print("Normal:    ", circ_in.count_ops())
    assert verify_equality(circ_out, circ_in)


if __name__ == '__main__':
    test()
