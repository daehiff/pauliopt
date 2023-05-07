import numpy as np
from pyzx import Mat2
from qiskit import QuantumCircuit


class CliffordTableau:
    def __init__(self, n_qubits: int = None, tableau: np.array = None):
        if n_qubits is None and tableau is None:
            raise Exception("Either Tableau or number of qubits must be defined")
        if tableau is None:
            self.tableau = np.eye(2 * n_qubits)
            self.n_qubits = n_qubits
        else:
            if tableau.shape[1] == tableau.shape[1]:
                raise Exception("Must be a 2nx2n Tableau!")
            self.n_qubits = int(tableau.shape[1] / 2.0)
            self.tableau = tableau
            if not 2 * self.n_qubits == tableau.shape[1]:
                raise Exception(f"Tableau of shape: {tableau.shape}, is not a factor of 2")

    def apply_h(self, qubit):
        self.tableau[:, [self.n_qubits + qubit, qubit]] = self.tableau[:, [qubit, self.n_qubits + qubit]]

    def apply_s(self, qubit):
        self.tableau[:, self.n_qubits + qubit] = (self.tableau[:, self.n_qubits + qubit] + self.tableau[:, qubit]) % 2

    def apply_cnot(self, control, target):
        self.tableau[:, target] = \
            (self.tableau[:, target] + self.tableau[:, control]) % 2
        self.tableau[:, self.n_qubits + control] = \
            (self.tableau[:, self.n_qubits + control] + self.tableau[:, self.n_qubits + target]) % 2

    def to_clifford_circuit(self):
        qc = QuantumCircuit(self.n_qubits)

        remaining = Mat2(self.tableau).inverse()  # TODO check for known row_add issues..:D

        def x_out(inp, out):
            pass

        def z_out(inp, out):
            pass

        n = self.n_qubits

        for col in range(n):
            pivot_row = None
            for row in range(col, n):
                pass
