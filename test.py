import networkx as nx
from pyzx import Mat2
from qiskit.circuit import Gate
from stim import Tableau
import pyzx as zx

from pauliopt.pauli.anneal import anneal
from pauliopt.pauli.clifford_gates import H, S, V, CX, CY, CZ
from pauliopt.pauli.clifford_tableau import CliffordTableau
from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import *
from pytket.extensions.pyzx import tk_to_pyzx, pyzx_to_tk
from pytket.extensions.qiskit.qiskit_convert import tk_to_qiskit, qiskit_to_tk
import numpy as np
import scipy as sc
import qiskit.quantum_info as qi

from pauliopt.pauli.utils import Pauli, _pauli_to_string
import stim
from qiskit.providers.fake_provider import FakeHanoi, FakeLima, FakeTokyo, FakeLima, \
    FakeLagos
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from qiskit.quantum_info import Clifford


def pyzx_to_qiskit(circ: zx.Circuit) -> QuantumCircuit:
    return tk_to_qiskit(pyzx_to_tk(circ))


def two_qubit_count(count_ops):
    count = 0
    count += count_ops["cx"] if "cx" in count_ops.keys() else 0
    count += count_ops["cy"] if "cy" in count_ops.keys() else 0
    count += count_ops["cz"] if "cz" in count_ops.keys() else 0
    return count


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

    pp = PauliPolynomial()
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


def reconstruct_tableau(tableau: stim.Tableau):
    xsx, xsz, zsx, zsz, x_signs, z_signs = tableau.to_numpy()
    x_row = np.concatenate([xsx, xsz], axis=1)
    z_row = np.concatenate([zsx, zsz], axis=1)
    return np.concatenate([x_row, z_row], axis=0).astype(np.int64)


def reconstruct_tableau_signs(tableau: stim.Tableau):
    xsx, xsz, zsx, zsz, x_signs, z_signs = tableau.to_numpy()
    signs = np.concatenate([x_signs, z_signs]).astype(np.int64)
    return signs


def circuit_to_tableau(circ: QuantumCircuit):
    tableau = stim.Tableau(circ.num_qubits)
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


def random_clifford_circuit(nr_gates=20, nr_qubits=4, gate_choice=None):
    qc = QuantumCircuit(nr_qubits)
    if gate_choice is None:
        gate_choice = ["CY", "CZ", "CX", "H", "S", "V"]
    for _ in range(nr_gates):
        gate_t = np.random.choice(gate_choice)
        if gate_t == "CX":
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
        elif gate_t == "H":
            qubit = np.random.choice([i for i in range(nr_qubits)])
            qc.h(qubit)
        elif gate_t == "S":
            qubit = np.random.choice([i for i in range(nr_qubits)])
            qc.s(qubit)
        elif gate_t == "V":
            qubit = np.random.choice([i for i in range(nr_qubits)])
            qc.sx(qubit)
        elif gate_t == "CX":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cx(control, target)
    return qc


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


def get_ops_count(qc: QuantumCircuit):
    count = {"single": 0, "two": 0}
    ops = qc.count_ops()
    if "cx" in ops.keys():
        count["two"] += ops["cx"]
    if "swap" in ops.keys():
        count["two"] += 3 * ops["swap"]
    if "h" in ops.keys():
        count["single"] += ops["h"]
    if "s" in ops.keys():
        count["single"] += ops["s"]
    if "x" in ops.keys():
        count["single"] += ops["x"]
    if "y" in ops.keys():
        count["single"] += ops["y"]
    if "z" in ops.keys():
        count["single"] += ops["z"]
    return count


def experiment(num_qubits=7):
    num_qubits = 7
    df = pd.DataFrame(
        columns=["n_rep", "num_qubits", "n_gadgets", "arch", "single", "two"])
    for n_gadgets in range(10, 250, 10):
        for _ in range(10):
            circ = random_hscx_circuit(nr_qubits=num_qubits, nr_gates=n_gadgets)
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "arch": "original"} \
                     | get_ops_count(circ)
            df.loc[len(df)] = column
            df.to_csv("test1.csv")
            for name, topo in [("complete", Topology.complete(num_qubits)),
                               ("line", Topology.line(num_qubits))]:
                ct = CliffordTableau.from_circuit(circ)
                circ_out = ct.to_cifford_circuit_arch_aware(topo)
                column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                          "arch": name} \
                         | get_ops_count(circ_out)
                df.loc[len(df)] = column
                df.to_csv("test1.csv")

            ct = CliffordTableau.from_circuit(circ)
            circ_out = ct.to_clifford_circuit()
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "arch": "complete_stim"} \
                     | get_ops_count(circ_out)
            df.loc[len(df)] = column
            df.to_csv("test1.csv")

            ct = Clifford(circ)
            circ_out = ct.to_circuit()
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "arch": "complete_qiskit"} \
                     | get_ops_count(circ_out)
            df.loc[len(df)] = column
            df.to_csv("test1.csv")

    df.to_csv("test1.csv")
    # print(circ_out)
    # print(circ_stim)

    assert (verify_equality(circ, circ_out))


def plot():
    df = pd.read_csv("test1.csv")
    # df = df[df.arch != "original"]
    df["single"] = df["single"] / 2.0

    sns.lineplot(df, x="n_gadgets", y="single", hue="arch")
    plt.title("Single Qubits")

    # plt.savefig("single.png")
    # plt.clf()
    plt.show()

    sns.lineplot(df, x="n_gadgets", y="two", hue="arch")
    plt.title("Two Qubits")
    # plt.savefig("two.png")
    # plt.clf()
    plt.show()


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


def main_():
    pp = generate_random_pauli_polynomial(8, 200)
    assert isinstance(pp, PauliPolynomial)

    pp_ = simplify_pauli_polynomial(pp)

    print(verify_equality(pp_.to_qiskit(), pp.to_qiskit()))


if __name__ == '__main__':
    main_()
