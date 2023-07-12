from abc import ABC, abstractmethod
from numbers import Number

from pauliopt.utils import AngleExpr


class Gate(ABC):
    def __init__(self, qubits):
        self.qubits = qubits

    @abstractmethod
    def to_qiskit(self):
        ...

    @abstractmethod
    def inverse(self):
        ...

    @abstractmethod
    def __repr__(self):
        ...


class SingleQubitGate(Gate, ABC):
    def __init__(self, qubit):
        super().__init__([qubit])


class H(SingleQubitGate):
    def __init__(self, qubit):
        super().__init__(qubit)

    def to_qiskit(self):
        from qiskit.circuit.library import HGate
        return HGate(), self.qubits

    def inverse(self):
        return H(self.qubits[0])

    def __repr__(self):
        return f"H({self.qubits[0]})"


class S(SingleQubitGate):
    def __init__(self, qubit):
        super().__init__(qubit)

    def to_qiskit(self):
        from qiskit.circuit.library import SGate
        return SGate(), self.qubits

    def inverse(self):
        return Sdg(self.qubits[0])

    def __repr__(self):
        return f"S({self.qubits[0]})"


class Sdg(SingleQubitGate):
    def __init__(self, qubit):
        super().__init__(qubit)

    def to_qiskit(self):
        from qiskit.circuit.library import SdgGate
        return SdgGate(), self.qubits

    def inverse(self):
        return S(self.qubits[0])

    def __repr__(self):
        return f"Sdg({self.qubits[0]})"


class V(SingleQubitGate):
    def __init__(self, qubit):
        super().__init__(qubit)

    def to_qiskit(self):
        from qiskit.circuit.library import SXGate
        return SXGate(), self.qubits

    def inverse(self):
        return Vdg(self.qubits[0])

    def __repr__(self):
        return f"V({self.qubits[0]})"


class Vdg(SingleQubitGate):
    def __init__(self, qubit):
        super().__init__(qubit)

    def to_qiskit(self):
        from qiskit.circuit.library import SXdgGate
        return SXdgGate(), self.qubits

    def inverse(self):
        return V(self.qubits[0])

    def __repr__(self):
        return f"Vdg({self.qubits[0]})"


class Rz(SingleQubitGate):
    def __init__(self, angle, qubit):
        super().__init__(qubit)
        self.angle = angle

    def to_qiskit(self):
        from qiskit.circuit.library import RZGate

        if isinstance(self.angle, Number):
            angle = self.angle
        elif isinstance(self.angle, AngleExpr):
            angle = self.angle.to_qiskit
        else:
            raise TypeError(
                f"Angle must either be float or AngleExpr, but got {type(self.angle)}")
        return RZGate(angle), self.qubits

    def inverse(self):
        return Rz(-self.angle, self.qubits[0])

    def __repr__(self):
        return f"Rz({self.qubits[0]}, {self.angle})"


class TwoQubitGate(Gate, ABC):
    def __init__(self, control, target):
        super().__init__([control, target])


class CX(TwoQubitGate):
    def __init__(self, control, target):
        super().__init__(control, target)

    def to_qiskit(self):
        from qiskit.circuit.library import CXGate
        return CXGate(), self.qubits

    def inverse(self):
        return CX(self.qubits[0], self.qubits[1])

    def __repr__(self):
        return f"CX({self.qubits[0]}, {self.qubits[1]})"


class PauliCircuit:

    def __init__(self, num_qubits):
        self.global_phase = 0
        self.num_qubits = num_qubits
        self.gates = []

    def compose(self, other: "PauliCircuit"):
        self.gates += other.gates
        return self

    def __add__(self, other: "PauliCircuit"):
        return self.compose(other)

    def __len__(self):
        return len(self.gates)

    def h(self, qubit):
        self.gates.append(H(qubit))
        return self

    def s(self, qubit):
        self.gates.append(S(qubit))
        return self

    def sdg(self, qubit):
        self.gates.append(Sdg(qubit))
        return self

    def v(self, qubit):
        self.gates.append(V(qubit))
        return self

    def vdg(self, qubit):
        self.gates.append(Vdg(qubit))
        return self

    def rz(self, angle, qubit):
        self.gates.append(Rz(angle, qubit))
        return self

    def cx(self, control, target):
        self.gates.append(CX(control, target))
        return self

    def __repr__(self):
        rep = ""
        for gate in self.gates:
            rep += f"{gate}\n"
        return rep

    def to_qiskit(self):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(self.num_qubits)
        qc.global_phase = self.global_phase
        for gate in self.gates:
            qiskit_gate, qubits = gate.to_qiskit()
            qc.append(qiskit_gate, qubits)
        return qc

    def inverse(self):
        inverse_circuit = PauliCircuit(self.num_qubits)
        for gate in reversed(self.gates):
            inverse_circuit.gates.append(gate.inverse())
        return inverse_circuit

    def apply_permutation(self, permutation: list):
        new_gates = []
        for gate in self.gates:
            if isinstance(gate, TwoQubitGate):
                control, target = gate.qubits
                control = permutation.index(control)
                target = permutation.index(target)
                gate = CX(control, target)
            new_gates.append(gate)
        self.gates = new_gates
