import numpy as np


class ParityMap:
    def __init__(self, n_qubits):
        try:
            # pylint: disable = import-outside-toplevel
            import galois
        except ModuleNotFoundError as _:
            raise ModuleNotFoundError("You must install the 'galois' library.")

        self.n_qubits = n_qubits

        self.matrix = galois.GF2(np.eye(n_qubits, dtype=np.uint8))

    def append_cnot(self, control, target):
        self.matrix[target, :] += self.matrix[control, :]

    def prepend_cnot(self, control, target):
        self.matrix[:, target] += self.matrix[control, :]

    def transpose(self):
        self.matrix = self.matrix.T
        return self

    def copy(self):
        new = ParityMap(self.n_qubits)
        new.matrix = self.matrix.copy()
        return new

    def append_circuit(self, circuit):
        # TODO add inital and final permutation
        for gate in circuit._gates:
            if gate.name == "CX":
                self.append_cnot(gate.control, gate.target)
            else:
                raise ValueError("Only CNOT gates are supported.")

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, key):
        return self.matrix[key]

    @staticmethod
    def from_circuit_append(circuit):
        """
        Construct a parity map from a circuit.

        :param circuit: The circuit to convert.
        :return: The parity map.
        """
        parity_map = ParityMap(circuit.n_qubits)
        for gate in circuit._gates:
            if gate.name == "CX":
                parity_map.append_cnot(gate.control, gate.target)

            else:
                raise ValueError("Only CNOT gates are supported.")
        return parity_map

    @staticmethod
    def from_circuit_prepend(circuit):
        """
        Construct a parity map from a circuit.

        :param circuit: The circuit to convert.
        :return: The parity map.
        """
        parity_map = ParityMap(circuit.n_qubits)
        for gate in circuit._gates:
            if gate.name == "CX":
                parity_map.prepend_cnot(gate.control, gate.target)

            else:
                raise ValueError("Only CNOT gates are supported.")
        return parity_map
