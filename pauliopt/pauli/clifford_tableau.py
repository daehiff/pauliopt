import numpy as np
from pyzx import Mat2
from qiskit import QuantumCircuit
import stim


def reconstruct_tableau(tableau: stim.Tableau):
    xsx, xsz, zsx, zsz, x_signs, z_signs = tableau.to_numpy()
    x_row = np.concatenate([xsx, xsz], axis=1)
    z_row = np.concatenate([zsx, zsz], axis=1)
    return np.concatenate([x_row, z_row], axis=0).astype(np.int64)


def reconstruct_tableau_signs(tableau: stim.Tableau):
    xsx, xsz, zsx, zsz, x_signs, z_signs = tableau.to_numpy()
    signs = np.concatenate([x_signs, z_signs]).astype(np.int64)
    return signs


class CliffordTableau:
    def __init__(self, n_qubits: int = None, tableau: np.array = None, signs: np.array = None):
        if n_qubits is None and tableau is None:
            raise Exception("Either Tableau or number of qubits must be defined")
        if tableau is None:
            self.tableau = np.eye(2 * n_qubits)
            self.signs = np.zeros((2 * n_qubits))
            self.n_qubits = n_qubits
        else:
            if tableau.shape[0] != tableau.shape[1]:
                raise Exception(
                    f"Must be a 2nx2n Tableau, but is of shape: {tableau.shape}")
            self.n_qubits = int(tableau.shape[1] / 2.0)
            self.tableau = tableau
            if signs is None:
                self.signs = np.zeros((2 * self.n_qubits))
            else:
                self.signs = signs
            if not 2 * self.n_qubits == tableau.shape[1]:
                raise Exception(
                    f"Tableau of shape: {tableau.shape}, is not a factor of 2")

    def apply_h(self, qubit):
        self.signs = (self.signs + self.tableau[:, qubit] * self.tableau[:, self.n_qubits + qubit]) % 2

        self.tableau[:, [self.n_qubits + qubit, qubit]] = self.tableau[:,
                                                          [qubit, self.n_qubits + qubit]]

    def apply_s(self, qubit):
        self.signs = (self.signs + self.tableau[:, qubit] * self.tableau[:, self.n_qubits + qubit]) % 2

        self.tableau[:, self.n_qubits + qubit] = (self.tableau[:, self.n_qubits + qubit] +
                                                  self.tableau[:, qubit]) % 2

    def apply_cnot(self, control, target):
        x_ia = self.tableau[:, control]
        x_ib = self.tableau[:, target]

        z_ia = self.tableau[:, self.n_qubits + control]
        z_ib = self.tableau[:, self.n_qubits + target]

        self.tableau[:, target] = \
            (self.tableau[:, target] + self.tableau[:, control]) % 2
        self.tableau[:, self.n_qubits + control] = \
            (self.tableau[:, self.n_qubits + control] + self.tableau[:,
                                                        self.n_qubits + target]) % 2
        tmp_sum = ((x_ib + z_ia) % 2 + np.ones(z_ia.shape)) % 2
        self.signs = (self.signs + x_ia * z_ib * tmp_sum) % 2

    @staticmethod
    def from_circuit(circ: QuantumCircuit):
        tableau = CliffordTableau(n_qubits=circ.num_qubits)
        for op in circ:
            if op.operation.name == "h":
                tableau.apply_h(op.qubits[0].index)
            elif op.operation.name == "s":
                tableau.apply_s(op.qubits[0].index)
            elif op.operation.name == "cx":
                tableau.apply_cnot(op.qubits[0].index, op.qubits[1].index)
        return tableau

    def inverse(self):

        x2x = self.tableau[:self.n_qubits, :self.n_qubits].copy().astype(np.bool8)
        z2z = self.tableau[self.n_qubits:2 * self.n_qubits, self.n_qubits:2 * self.n_qubits].copy().astype(np.bool8)
        x2z = self.tableau[:self.n_qubits, self.n_qubits:2 * self.n_qubits].copy().astype(np.bool8)
        z2x = self.tableau[self.n_qubits:2 * self.n_qubits, :self.n_qubits].copy().astype(np.bool8)
        tab = stim.Tableau.from_numpy(x2x=x2x,
                                      x2z=x2z,
                                      z2x=z2x,
                                      z2z=z2z,
                                      x_signs=self.signs[0:self.n_qubits].copy().astype(np.bool8),
                                      z_signs=self.signs[self.n_qubits:2 * self.n_qubits].copy().astype(np.bool8))
        t_inv = tab.inverse()
        ct_inv = reconstruct_tableau(t_inv)
        ct_signs = reconstruct_tableau_signs(t_inv)
        # x_row = np.concatenate([xsx, xsz], axis=1)
        # z_row = np.concatenate([zsx, zsz], axis=1)
        # inv_tableau = np.concatenate([x_row, z_row], axis=0).astype(np.int64)
        # p_string = np.zeros((2 * self.n_qubits))
        # p_string[1] = 1
        # print((inv_tableau @ (self.tableau @ ((self.signs * p_string) % 2) % 2)) % 2)

        return CliffordTableau(tableau=ct_inv, signs=ct_signs)

    def print_zx(self):
        def x_out(inp, out):
            return self.tableau[inp, out] + \
                   2 * self.tableau[inp, out + self.n_qubits]

        def z_out(inp, out):
            return self.tableau[inp + self.n_qubits, out] + \
                   2 * self.tableau[inp + self.n_qubits, out + self.n_qubits]

        print("X: ")
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                print(f" {x_out(i, j)} ", end="")
            print()
        print("Z: ")
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                print(f" {z_out(i, j)} ", end="")
            print()

    def to_clifford_circuit(self):
        qc = QuantumCircuit(self.n_qubits)

        remaining = self.inverse()

        def apply(gate_name: str, gate_data: tuple):
            if gate_name == "CNOT":
                remaining.apply_cnot(gate_data[0], gate_data[1])
                qc.cx(gate_data[0], gate_data[1])
            elif gate_name == "H":
                remaining.apply_h(gate_data[0])
                qc.h(gate_data[0])
            elif gate_name == "S":
                remaining.apply_s(gate_data[0])
                qc.s(gate_data[0])
            else:
                raise Exception("Unknown Gate")

        def x_out(inp, out):
            return remaining.tableau[inp, out] + \
                   2 * remaining.tableau[inp, out + self.n_qubits]

        def z_out(inp, out):
            return remaining.tableau[inp + self.n_qubits, out] + \
                   2 * remaining.tableau[inp + self.n_qubits, out + self.n_qubits]

        n = self.n_qubits
        for col in range(n):
            pivot_row = n - 1
            for row in range(col, n):
                px = x_out(col, row)
                pz = z_out(col, row)
                if px != 0 and pz != 0 and px != pz:
                    pivot_row = row
                    break
            assert pivot_row is not None
            # Move the pivot to the diagonal
            if pivot_row != col:
                apply("CNOT", (pivot_row, col))
                apply("CNOT", (col, pivot_row))
                apply("CNOT", (pivot_row, col))

            # Transform the pivot to the XZ plane
            if z_out(col, col) == 3:
                apply("S", (col,))

            if z_out(col, col) != 2:
                apply("H", (col,))

            if x_out(col, col) != 1:
                apply("S", (col,))

            # Use the pivot to remove all other terms in the X observable.
            for row in range(col + 1, n):
                if x_out(col, row) == 3:
                    apply("S", (row,))

            for row in range(col + 1, n):
                if x_out(col, row) == 2:
                    apply("H", (row,))

            for row in range(col + 1, n):
                if x_out(col, row) != 0:
                    apply("CNOT", (col, row))

            # Use the pivot to remove all other terms in the Z observable.
            for row in range(col + 1, n):
                if z_out(col, row) == 3:
                    apply("S", (row,))

            for row in range(col + 1, n):
                if z_out(col, row) == 1:
                    apply("H", (row,))

            for row in range(col + 1, n):
                if z_out(col, row) != 0:
                    apply("CNOT", (row, col))

        signs_copy_z = remaining.signs[n:2 * n].copy()
        for col in range(n):
            if signs_copy_z[col] != 0:
                apply("H", (col,))

        for col in range(n):
            if signs_copy_z[col] != 0:
                apply("S", (col,))
                apply("S", (col,))

        for col in range(n):
            if signs_copy_z[col] != 0:
                apply("H", (col,))
        for col in range(n):
            if remaining.signs[col] != 0:
                apply("S", (col,))
                apply("S", (col,))

        return qc
