from math import pi

from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.random import random_circuit

from pauliopt.clifford.tableau import CliffordTableau


def get_qubits(qubits, qreg):
    qubits_ = []
    for qubit in qubits:
        qubits_.append(qreg.index(qubit))

    return tuple(qubits_)


def main():
    ct = CliffordTableau(3)

    ct.append_h(0)
    ct.append_cnot(0, 1)
    ct.append_s(1)

    from pauliopt.topologies import Topology

    topology = Topology.complete(4)

    from pauliopt.clifford.tableau_synthesis import synthesize_tableau

    qc, perm = synthesize_tableau(ct, topology, include_swaps=False)


if __name__ == "__main__":
    main()
