import numpy as np
from qiskit import QuantumCircuit

from pauliopt.circuits import Circuit
from pauliopt.cnots.parity_map import ParityMap
from pauliopt.cnots.synthesis.perm_row_col import _perm_row_col
from pauliopt.combs.synthesis.cx_comb_synthesis import comb_row_col
from pauliopt.topologies import Topology
from tests.utils import verify_equality


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


def main():
    qc = QuantumCircuit(3)

    qc.cx(0, 1)
    qc.cx(1, 2)

    qc.h(0)
    qc.h(1)

    qc.cx(0, 1)
    qc.cx(2, 1)

    topo = Topology.line(3)

    circ = Circuit.from_qiskit(qc)
    circ_ = Circuit.from_qiskit(qc)

    circ_out = comb_row_col(circ, topo)

    circ_out = circ_out.to_qiskit(include_permutations=False)

    print(circ_out)
    print(qc)

    print(verify_equality(qc, circ_out))

    # comb = circuit_to_cnot_comb(circ)


if __name__ == "__main__":
    main()
