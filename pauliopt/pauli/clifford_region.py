from pauliopt.pauli.clifford_gates import *
from pauliopt.pauli.clifford_tableau import CliffordTableau
from pauliopt.topologies import Topology


class CliffordRegion:
    def __init__(self, num_qubits, gates=None):
        if gates is None:
            gates = []
        self.gates: [CliffordGate] = gates
        self.num_qubits = num_qubits

    def append_gate(self, gate: CliffordGate):
        if isinstance(gate, SingleQubitGate) and gate.qubit >= self.num_qubits:
            raise Exception(
                f"Gate with {gate.qubit} is out of bounds for Clifford Region with Qubits: {self.num_qubits}")
        if isinstance(gate,
                      ControlGate) and gate.control >= self.num_qubits and gate.target >= self.num_qubits:
            raise Exception(
                f"Control Gate  with {gate.control}, {gate.target} is out of bounds for Clifford Region with Qubits: {self.num_qubits}")
        self.gates.append(gate)

    def prepend_gate(self, gate: CliffordGate):
        if isinstance(gate, SingleQubitGate) and gate.qubit >= self.num_qubits:
            raise Exception(
                f"Gate with {gate.qubit} is out of bounds for Clifford Region with Qubits: {self.num_qubits}")
        if isinstance(gate,
                      ControlGate) and gate.control >= self.num_qubits and gate.target >= self.num_qubits:
            raise Exception(
                f"Control Gate  with {gate.control}, {gate.target} is out of bounds for Clifford Region with Qubits: {self.num_qubits}")
        self.gates.insert(0, gate)

    def to_qiskit(self, method, topology: Topology = None, include_swaps=False):
        if method == "ct_resynthesis":
            ct = CliffordTableau(self.num_qubits)
            for gate in self.gates:
                ct.append_gate(gate)
            return ct.to_cifford_circuit_arch_aware(topology, include_swaps=include_swaps)
        else:
            raise NotImplementedError(f"Method {method} not implemented")
