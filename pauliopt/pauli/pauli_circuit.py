from abc import ABC, abstractmethod
from enum import Enum
from numbers import Number

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import Permutation

from pauliopt.utils import AngleExpr
from pauliopt.pauli.utils import X, Y, Z, I


class CliffordType(Enum):
    CX = "cx"
    CY = "cy"
    CZ = "cz"
    H = "h"
    S = "s"
    Sdg = "sdg"
    V = "v"
    Vdg = "vdg"


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


class CliffordGate(ABC):
    clifford_type = None
    rules = None

    def __init__(self):
        pass

    @abstractmethod
    def propagate_pauli(self, gadget: "pauliopt.pauli.pauli_gadget.PauliGadget"):
        ...


class Barrier(Gate):
    def __init__(self, num_qubits):
        super().__init__([])
        self.num_qubits = num_qubits

    def to_qiskit(self):
        from qiskit.circuit.library import Barrier as Barrier_Qiskit

        return Barrier_Qiskit(self.num_qubits), None

    def inverse(self):
        return Barrier(self.num_qubits)

    def __repr__(self):
        return f"Barrier()"


class SingleQubitGate(Gate, ABC):
    def __init__(self, qubit):
        super().__init__([qubit])


class SingleQubitClifford(Gate, CliffordGate, ABC):
    def __init__(self, qubit):
        super().__init__([qubit])

    @property
    def qubit(self):
        return self.qubits[0]

    def propagate_pauli(self, gadget: "pauliopt.pauli.pauli_gadget.PauliGadget"):
        if self.rules is None:
            raise Exception(f"{self} has no rules defined for propagation!")
        p_string = gadget.paulis[self.qubits[0]].value
        new_p, phase_change = self.rules[p_string]
        gadget.paulis[self.qubits[0]] = new_p
        if phase_change == -1:
            gadget.angle *= phase_change
        return gadget


class H(SingleQubitClifford):
    rules = {"X": (Z, 1), "Y": (Y, -1), "Z": (X, 1), "I": (I, 1)}
    clifford_type = CliffordType.H

    def __init__(self, qubit):
        super().__init__(qubit)

    def to_qiskit(self):
        from qiskit.circuit.library import HGate

        return HGate(), self.qubits

    def inverse(self):
        return H(self.qubits[0])

    def __repr__(self):
        return f"H({self.qubits[0]})"


class S(SingleQubitClifford):
    rules = {"X": (Y, -1), "Y": (X, 1), "Z": (Z, 1), "I": (I, 1)}
    clifford_type = CliffordType.S

    def __init__(self, qubit):
        super().__init__(qubit)

    def to_qiskit(self):
        from qiskit.circuit.library import SGate

        return SGate(), self.qubits

    def inverse(self):
        return Sdg(self.qubits[0])

    def __repr__(self):
        return f"S({self.qubits[0]})"


class Sdg(SingleQubitClifford):
    rules = {"X": (Y, 1), "Y": (X, -1), "Z": (Z, 1), "I": (I, 1)}
    clifford_type = CliffordType.Sdg

    def __init__(self, qubit):
        super().__init__(qubit)

    def to_qiskit(self):
        from qiskit.circuit.library import SdgGate

        return SdgGate(), self.qubits

    def inverse(self):
        return S(self.qubits[0])

    def __repr__(self):
        return f"Sdg({self.qubits[0]})"


class V(SingleQubitClifford):
    rules = {"X": (X, 1), "Y": (Z, -1), "Z": (Y, 1), "I": (I, 1)}
    clifford_type = CliffordType.V

    def __init__(self, qubit):
        super().__init__(qubit)

    def to_qiskit(self):
        from qiskit.circuit.library import SXGate

        return SXGate(), self.qubits

    def inverse(self):
        return Vdg(self.qubits[0])

    def __repr__(self):
        return f"V({self.qubits[0]})"


class Vdg(SingleQubitClifford):
    rules = {"X": (X, 1), "Y": (Z, 1), "Z": (Y, -1), "I": (I, 1)}
    clifford_type = CliffordType.Vdg

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
            angle = float(self.angle)
        elif isinstance(self.angle, AngleExpr):
            angle = self.angle.to_qiskit
        else:
            raise TypeError(
                f"Angle must either be float or AngleExpr, but got {type(self.angle)}"
            )
        return RZGate(angle), self.qubits

    def inverse(self):
        return Rz(-self.angle, self.qubits[0])

    def __repr__(self):
        return f"Rz({self.qubits[0]}, {self.angle})"


class TwoQubitGate(Gate, ABC):
    def __init__(self, control, target):
        super().__init__([control, target])


class TwoQubitClifford(Gate, CliffordGate, ABC):
    def __init__(self, control, target):
        super().__init__([control, target])

    @property
    def control(self):
        return self.qubits[0]

    @property
    def target(self):
        return self.qubits[1]

    def propagate_pauli(self, gadget: "pauliopt.pauli.pauli_gadget.PauliGadget"):
        if self.rules is None:
            raise Exception(f"{self} has no rules defined for propagation!")
        pauli_size = len(gadget)
        if self.control >= pauli_size or self.target >= pauli_size:
            raise Exception(
                f"Control: {self.control} or Target {self.target} out of bounds: {pauli_size}"
            )
        p_string = gadget.paulis[self.control].value + gadget.paulis[self.target].value
        p_c, p_t, phase_change = self.rules[p_string]
        gadget.paulis[self.control] = p_c
        gadget.paulis[self.target] = p_t
        if phase_change == -1:
            gadget.angle *= phase_change
        return gadget


class CX(TwoQubitClifford):
    rules = {
        "XX": (X, I, 1),
        "XY": (Y, Z, 1),
        "XZ": (Y, Y, -1),
        "XI": (X, X, 1),
        "YX": (Y, I, 1),
        "YY": (X, Z, -1),
        "YZ": (X, Y, 1),
        "YI": (Y, X, 1),
        "ZX": (Z, X, 1),
        "ZY": (I, Y, 1),
        "ZZ": (I, Z, 1),
        "ZI": (Z, I, 1),
        "IX": (I, X, 1),
        "IY": (Z, Y, 1),
        "IZ": (Z, Z, 1),
        "II": (I, I, 1),
    }
    clifford_type = CliffordType.CX

    def __init__(self, control, target):
        super().__init__(control, target)

    def to_qiskit(self):
        from qiskit.circuit.library import CXGate

        return CXGate(), self.qubits

    def inverse(self):
        return CX(self.qubits[0], self.qubits[1])

    def __repr__(self):
        return f"CX({self.qubits[0]}, {self.qubits[1]})"


class CY(TwoQubitClifford):
    rules = {
        "XX": (Y, Z, -1),
        "XY": (X, I, 1),
        "XZ": (Y, X, 1),
        "XI": (X, Y, 1),
        "YX": (X, Z, 1),
        "YY": (Y, I, 1),
        "YZ": (X, X, -1),
        "YI": (Y, Y, 1),
        "ZX": (I, X, 1),
        "ZY": (Z, Y, 1),
        "ZZ": (I, Z, 1),
        "ZI": (Z, I, 1),
        "IX": (Z, X, 1),
        "IY": (I, Y, 1),
        "IZ": (Z, Z, 1),
        "II": (I, I, 1),
    }
    clifford_type = CliffordType.CY

    def __init__(self, control, target):
        super().__init__(control, target)

    def to_qiskit(self):
        from qiskit.circuit.library import CYGate

        return CYGate(), self.qubits

    def inverse(self):
        return CY(self.qubits[0], self.qubits[1])

    def __repr__(self):
        return f"CY({self.qubits[0]}, {self.qubits[1]})"


class CZ(TwoQubitClifford):
    rules = {
        "XX": (Y, Y, 1),
        "XY": (Y, X, -1),
        "XZ": (X, I, 1),
        "XI": (X, Z, 1),
        "YX": (X, Y, -1),
        "YY": (X, X, 1),
        "YZ": (Y, I, 1),
        "YI": (Y, Z, 1),
        "ZX": (I, X, 1),
        "ZY": (I, Y, 1),
        "ZZ": (Z, Z, 1),
        "ZI": (Z, I, 1),
        "IX": (Z, X, 1),
        "IY": (Z, Y, 1),
        "IZ": (I, Z, 1),
        "II": (I, I, 1),
    }
    clifford_type = CliffordType.CZ

    def __init__(self, control, target):
        super().__init__(control, target)

    def to_qiskit(self):
        from qiskit.circuit.library import CZGate

        return CZGate(), self.qubits

    def inverse(self):
        return CZ(self.qubits[0], self.qubits[1])

    def __repr__(self):
        return f"CZ({self.qubits[0]}, {self.qubits[1]})"


class PauliCircuit:
    def __init__(self, num_qubits):
        self.global_phase = 0
        self.num_qubits = num_qubits
        self.final_permutation = None
        self.gates = []

    def compose(self, other: "PauliCircuit"):
        self.gates += other.gates
        return self

    def __add__(self, other: "PauliCircuit"):
        return self.compose(other)

    @property
    def perm_gates(self):
        if self.final_permutation is None:
            return []

        perm = Permutation(self.num_qubits, self.final_permutation)

        gates = []
        qc = perm.decompose(gates_to_decompose=["cx"])
        qc = transpile(qc, basis_gates=["cx"])
        qreg = qc.qregs[0]
        for inst in qc:
            print(inst.operation.name)
            assert inst.operation.name == "cx"

            qubits = [qreg.index(q) for q in inst.qubits]
            gates.append(CX(qubits[0], qubits[1]))
        return gates

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

    def barrier(self, num_qubits=None):
        if num_qubits is None:
            self.gates.append(Barrier(self.num_qubits))
        else:
            self.gates.append(Barrier(num_qubits))
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
            # TODO quick and dirty fix for barriers
            if isinstance(gate, Barrier):
                qc.barrier()
            else:
                qiskit_gate, qubits = gate.to_qiskit()
                qc.append(qiskit_gate, qubits)

        if self.final_permutation is not None:
            qc.compose(
                Permutation(self.num_qubits, self.final_permutation), inplace=True
            )
        return qc

    def inverse(self):
        inverse_circuit = PauliCircuit(self.num_qubits)
        for gate in reversed(self.gates):
            inverse_circuit.gates.append(gate.inverse())
        return inverse_circuit

    def reverse_ops(self):
        p_circ = PauliCircuit(self.num_qubits)
        p_circ.gates = list(reversed(self.gates))
        return self

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


def generate_random_clifford(c_type: CliffordType, n_qubits: int):
    qubit = np.random.choice(list(range(n_qubits)))
    if c_type == CliffordType.CX:
        control = np.random.choice([i for i in range(n_qubits) if i != qubit])
        return CX(control, qubit)
    elif c_type == CliffordType.CY:
        control = np.random.choice([i for i in range(n_qubits) if i != qubit])
        return CY(control, qubit)
    elif c_type == CliffordType.CZ:
        control = np.random.choice([i for i in range(n_qubits) if i != qubit])
        return CZ(control, qubit)
    elif c_type == CliffordType.H:
        return H(qubit)
    elif c_type == CliffordType.S:
        return S(qubit)
    elif c_type == CliffordType.Sdg:
        return Sdg(qubit)
    elif c_type == CliffordType.Vdg:
        return Vdg(qubit)
    elif c_type == CliffordType.V:
        return V(qubit)

    else:
        raise TypeError(f"Unknown Clifford Type: {c_type}")
