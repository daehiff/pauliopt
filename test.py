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
    xsx, xsz, zsx, zsz, _, _ = tableau.to_numpy()
    x_row = np.concatenate([xsx, xsz], axis=1)
    z_row = np.concatenate([zsx, zsz], axis=1)
    return np.concatenate([x_row, z_row], axis=0).astype(np.int64)


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


def main(num_qubits=3):
    circ = QuantumCircuit(4)
    circ.s(1)
    circ.s(3)
    circ.cx(0, 3)
    circ.s(1)
    circ.cx(3, 2)
    circ.h(1)
    circ.s(2)
    circ.cx(1, 0)
    circ.h(1)
    circ.cx(3, 0)
    # circ = random_hscx_circuit(nr_gates=10)
    print(circ)

    ct = CliffordTableau.from_circuit(circ)
    print(ct.tableau)
    circ_out = ct.to_clifford_circuit()
    print(circ_out)
    # qc_out = cl_tableau.to_clifford_circuit()
    print(verify_equality(circ, circ_out))


def create_rules_graph(rules):
    rules_tuple = []
    for k, v in rules.items():
        res_key = _pauli_to_string(v[0]) + _pauli_to_string(v[1])
        rules_tuple.append((k, res_key))
    G = nx.Graph()
    print(rules_tuple)
    G.add_edges_from(rules_tuple)
    return G


if __name__ == '__main__':
    main()
