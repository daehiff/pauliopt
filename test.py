import networkx as nx
from pyzx import Mat2
from stim import Tableau
import pyzx as zx

from pauliopt.pauli.clifford_tableau import CliffordTableau
from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import *
from pytket.extensions.pyzx import tk_to_pyzx, pyzx_to_tk
from pytket.extensions.qiskit.qiskit_convert import tk_to_qiskit, qiskit_to_tk
import numpy as np

from pauliopt.pauli.utils import Pauli, _pauli_to_string
import stim
from qiskit.providers.fake_provider import FakeHanoi, FakeLima, FakeTokyo, FakeLima, FakeLagos


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


def compute_reduction_order(topo: Topology):
    pass


def main(num_qubits=3):
    dict = {
        "num_qubits": 7,
        "couplings": sorted([(0, 1), (0, 4), (1, 2), (1, 3), (4, 5), (4, 6)])
    }
    backend = np.random.choice([FakeLima(), FakeLagos()])
    topo = Topology.from_qiskit_backend(backend)
    print(topo.num_qubits)
    # circ = QuantumCircuit(4)
    # circ.s(1)
    # circ.s(3)
    # circ.cx(0, 3)
    # circ.s(1)
    # circ.cx(3, 2)
    # circ.h(1)
    # circ.s(2)
    # circ.cx(1, 0)
    # circ.h(1)
    # circ.cx(3, 0)
    # circ = QuantumCircuit(4)
    # circ.s(0)
    # circ.cx(1, 2)
    # circ.cx(2, 0)
    # circ.cx(2, 0)
    # circ.cx(2, 1)
    # circ.cx(1, 3)
    # circ.cx(0, 2)
    # circ.h(3)
    # circ.h(3)
    # circ.s(0)
    # circ.s(1)
    # circ.h(0)
    # circ.cx(0, 2)
    # circ.s(0)
    # circ.cx(2, 1)
    # circ.s(0)
    # circ.h(1)
    # circ.s(2)
    # circ.cx(0, 3)
    # circ.h(3)
    circ = random_hscx_circuit(nr_qubits=topo.num_qubits, nr_gates=400)
    circ.qasm(filename="test6.qasm")
    # circ = QuantumCircuit.from_qasm_file("test5.qasm")

    ct = CliffordTableau.from_circuit(circ)
    circ_out = ct.to_cifford_circuit_arch_aware(topo)
    # print(circ_out)
    # print(circ_stim)

    assert (verify_equality(circ, circ_out))


if __name__ == '__main__':
    for _ in range(100):
        main()
