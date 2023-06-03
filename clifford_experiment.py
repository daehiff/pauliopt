import itertools

import pyzx
from qiskit.circuit import Gate
import pyzx as zx

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyzx as zx
import seaborn as sns
import stim
from pytket.extensions.pyzx import pyzx_to_tk, tk_to_pyzx
from pytket.extensions.qiskit.qiskit_convert import tk_to_qiskit, qiskit_to_tk
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.providers.fake_provider import FakeVigo, FakeMumbai, FakeSherbrooke
from qiskit.quantum_info import Clifford

from pauliopt.pauli.anneal import anneal
from pauliopt.pauli.clifford_tableau import CliffordTableau
from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import *
from pauliopt.pauli.utils import X, Y, Z, I, Pauli
from pauliopt.utils import pi


def pyzx_to_qiskit(circ: pyzx.Circuit):
    return tk_to_qiskit(pyzx_to_tk(circ))


def qiskit_to_pyzx(circ: QuantumCircuit):
    return tk_to_pyzx(qiskit_to_tk(circ))


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
    gate_choice = ["H", "CX"]
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
    if "cz" in ops.keys():
        count["two"] += ops["cz"]

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
    # TODO check if all rz are cliffords
    if "sx" in ops.keys():
        count["single"] += ops["sx"]
    if "rz" in ops.keys():
        count["single"] += ops["rz"]
    return count


def experiment(num_qubits=7):
    backend = FakeMumbai()
    num_qubits = backend.configuration().num_qubits
    print(num_qubits)
    df = pd.DataFrame(
        columns=["n_rep", "num_qubits", "n_gadgets", "arch", "single", "two"])
    for n_gadgets in range(10, 300, 10):  # range(10, 150, 5)
        print(n_gadgets)
        for _ in range(10):
            print(_)
            # normal no optimization
            circ = random_hscx_circuit(nr_qubits=num_qubits, nr_gates=n_gadgets)
            circ_out = transpile(circ, backend=backend,
                                 basis_gates=["h", "s", "cx"],
                                 approximation_degree=1.0,
                                 initial_layout=
                                 [q for q in range(backend.configuration().num_qubits)])
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "arch": "original"} \
                     | get_ops_count(circ_out)
            df.loc[len(df)] = column
            df.to_csv("test1.csv")
            print("Default: ", circ_out.count_ops())

            # directly compilation to architecture
            topo = Topology.from_qiskit_backend(backend)
            ct = CliffordTableau.from_circuit(circ)
            circ_out = ct.to_cifford_circuit_arch_aware(topo)
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "arch": "tableau"} \
                     | get_ops_count(circ_out)
            df.loc[len(df)] = column
            df.to_csv("test1.csv")
            print("Tableau: ", circ_out.count_ops())

            # conversion to qiskit Clifford and back
            ct = Clifford(circ)
            circ_out = ct.to_circuit()
            circ_out = transpile(circ_out, backend=backend, basis_gates=["cx", "h", "s"])
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "arch": "qiskit"} \
                     | get_ops_count(circ_out)
            df.loc[len(df)] = column
            df.to_csv("test1.csv")

            # optimization with pyzx
            # circ_out = qiskit_to_pyzx(circ)
            # G = circ_out.to_graph()
            # pyzx.simplify.clifford_simp(G)
            # circ_out = pyzx.extract_circuit(G.copy()).to_basic_gates()
            # circ_out = pyzx_to_qiskit(circ_out)
            # circ_out = transpile(circ_out, backend=backend, optimization_level=0)
            # print("pyzx: ", circ_out.count_ops())
            # column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
            #           "arch": "pyzx"} \
            #          | get_ops_count(circ_out)
            # df.loc[len(df)] = column
            # df.to_csv("test1.csv")

    df.to_csv("test1.csv")
    # print(circ_out)
    # print(circ_stim)


def plot():
    nr_qubits = 27
    df = pd.read_csv("test1.csv")
    # df = df[df.arch != "original"]
    #
    # sns.lineplot(df, x="n_gadgets", y="single", hue="arch")
    # plt.title("Single Qubits")
    #
    # # plt.savefig("single.png")
    # # plt.clf()
    # plt.show()

    sns.lineplot(df, x="n_gadgets", y="two", hue="arch")
    # plot a horizontal line at nr_qubits**2/np.log2(nr_qubits)
    plt.axhline(y=nr_qubits ** 2)

    plt.title("Two Qubits")
    # plt.savefig("two.png")
    # plt.clf()
    plt.show()


if __name__ == "__main__":
    experiment()
    plot()
