from .clifford_gates import CliffordGate
from .pauli_gadget import PauliGadget

from qiskit import QuantumCircuit
import numpy as np
from pauliopt.topologies import Topology


class PauliPolynomial:
    def __init__(self):
        self.pauli_gadgets = []

    def __irshift__(self, gadget: PauliGadget):
        self.pauli_gadgets.append(gadget)
        return self

    def __rshift__(self, pauli_polynomial):
        for gadget in pauli_polynomial.pauli_gadgets:
            self.pauli_gadgets.append(gadget)
        return self

    def __repr__(self):
        rep_ = ""
        for gadet in self.pauli_gadgets:
            rep_ += str(gadet) + "\n"
        return rep_

    def __len__(self):
        return self.size

    @property
    def size(self):
        return len(self.pauli_gadgets)

    @property
    def num_qubits(self):
        return max([len(gadget) for gadget in self.pauli_gadgets])

    def to_qiskit(self, topology=None):
        num_qubits = self.num_qubits
        if topology is None:
            topology = Topology.complete(num_qubits)

        qc = QuantumCircuit(num_qubits)
        for gadget in self.pauli_gadgets:
            qc.compose(gadget.to_qiskit(topology), inplace=True)

        return qc

    def propagate(self, gate: CliffordGate):
        pp_ = PauliPolynomial()
        for gadget in self.pauli_gadgets:
            pp_ >>= gate.propagate_pauli(gadget)
        return pp_

    def copy(self):
        pp_ = PauliPolynomial()
        for gadget in self.pauli_gadgets:
            pp_ >>= gadget.copy()
        return pp_

    def two_qubit_count(self, topology, leg_chache=None):
        if leg_chache is None:
            leg_chache = {}
        count = 0
        for gadget in self.pauli_gadgets:
            count += gadget.two_qubit_count(topology, leg_chache=leg_chache)
        return count


def remove_collapsed_pauli_gadegts(remaining_poly):
    return list(filter(lambda x: x.angle != 2 * np.pi or x.angle != 0, remaining_poly))


def find_machting_parity_right(idx, remaining_poly):
    gadget = remaining_poly[idx]
    for idx_right, gadget_right in enumerate(remaining_poly[idx + 1:]):
        if np.all([p_1 == p_2 for p_1, p_2 in zip(gadget.paulis, gadget_right.paulis)]):
            return idx + idx_right
    return None


def is_commuting_region(idx, idx_right, remaining_poly):
    for k in range(idx, idx_right):
        if not remaining_poly[idx].commutes(remaining_poly[k]):
            return False
    return True


def propagate_phase_gadegts(remaining_poly):
    converged = True
    for idx, gadget in enumerate(remaining_poly):
        idx_right = find_machting_parity_right(idx, remaining_poly)
        if idx_right is None:
            continue
        if not is_commuting_region(idx, idx_right, remaining_poly):
            continue
        del remaining_poly[idx]
        converged = False
    return converged


def clamp(phase):
    new_phase = phase % 2
    if new_phase > 1:
        return new_phase - 2
    return phase


def simplify_pauli_polynomial(pp: PauliPolynomial):
    remaining_poly = [gadet for gadet in pp.pauli_gadgets]
    converged = False
    while not converged:
        remaining_poly = remove_collapsed_pauli_gadegts(remaining_poly)
        converged = propagate_phase_gadegts(remaining_poly)

    pp_ = PauliPolynomial()
    for gadget in remaining_poly:
        pp_ >>= gadget
    return pp_
