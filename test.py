import networkx as nx
from pyzx import Mat2
from stim import Tableau

from pauliopt.pauli.clifford_tableau import CliffordTableau
from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import *
import numpy as np

from pauliopt.pauli.utils import Pauli, _pauli_to_string
import stim


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


def main(num_qubits=3):
    # cl_tableau = CliffordTableau(2)
    # # cl_tableau.apply_h(0)
    # print(cl_tableau.tableau)
    # print("H")
    # cl_tableau = CliffordTableau(2)
    # cl_tableau.apply_h(0)
    # print(cl_tableau.tableau)
    # print("S")
    # cl_tableau = CliffordTableau(2)
    # cl_tableau.apply_s(0)
    # print(cl_tableau.tableau)
    # print("CX")
    cl_tableau = CliffordTableau(2)
    cl_tableau.apply_cnot(0, 1)
    print(cl_tableau.tableau)

    table = stim.Tableau(2)
    cnot = stim.Tableau.from_named_gate("CNOT")
    had = stim.Tableau.from_named_gate("H")
    table.append(had, [0])
    table.append(cnot, [0, 1])
    print(Mat2(reconstruct_tableau(table)).inverse())
    print(Mat2(reconstruct_tableau(table)))
    print("==")
    m = Mat2(reconstruct_tableau(table))
    m.col_add(0, 1)
    print(m)
    print(reconstruct_tableau(table.inverse(unsigned=True)))


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
