from pauliopt.cnots.parity_map import ParityMap


class CNOTComb(ParityMap):
    def __init__(
        self,
        n_qubits,
        holes,
        new_to_old_qubit_mappings,
        qubit_dependence,
        hole_plugs,
    ):
        super().__init__(n_qubits)
        self.n_qubits = n_qubits
        self.holes = holes
        self.new_to_old_qubit_mappings = new_to_old_qubit_mappings
        self.qubit_dependence = qubit_dependence
        self.hole_plugs = hole_plugs
