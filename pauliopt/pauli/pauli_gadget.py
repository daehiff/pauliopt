from collections import deque
from typing import List, Set, Union

import networkx as nx
import numpy as np

from pauliopt.pauli.utils import Pauli, X, Y, Z, I
from pauliopt.topologies import Topology
from pauliopt.utils import AngleExpr


def decompose_cnot_ladder_z(ctrl: int, trg: int, arch: Topology):
    """
    Helper function to route a CNOT between two qubits on a given architecture
    """
    cnot_ladder = []
    shortest_path = arch.shortest_path(ctrl, trg)

    prev = ctrl
    for current in shortest_path[1:-1]:
        cnot_ladder.append((current, prev))
        cnot_ladder.append((prev, current))
        prev = current
    cnot_ladder.append((shortest_path[-2], trg))
    return reversed(cnot_ladder)


def find_minimal_cx_assignment(column: np.array, arch: Topology):
    """
    Find a minimal CNOT assignment for a given column (which is a binary array) on a given architecture.
    :param column: Binary array
    :param arch: Architecture
    """
    if not np.all(np.isin(column, [0, 1])):
        raise Exception(f"Expected binary array as column, got: {column}")

    G = nx.Graph()
    for i in range(len(column)):
        G.add_node(i)

    for i in range(len(column)):
        for j in range(len(column)):
            if column[i] != 0 and column[j] != 0 and i != j:
                G.add_edge(i, j, weight=4 * arch.dist(i, j) - 2)

    mst_branches = list(nx.minimum_spanning_edges(G, data=False, algorithm="prim"))
    incident = {q: set() for q in range(len(column))}
    for fst, snd in mst_branches:
        incident[fst].add((fst, snd))
        incident[snd].add((snd, fst))

    q0 = np.argmax(column)  # Assume that 0 is always the first qubit aka first non zero
    visited = set()
    queue = deque([q0])
    cnot_ladder = []
    while queue:
        q = queue.popleft()
        visited.add(q)
        for tail, head in incident[q]:
            if head not in visited:
                cnot_ladder += decompose_cnot_ladder_z(head, tail, arch)
                queue.append(head)
    return cnot_ladder, q0


class PPhase:
    """
    Class to create a PauliGadget with a given angle:

    Example:
        >>> from pauliopt.pauli.utils import Pauli, X, Y, Z, I
        >>> from pauliopt.pauli.pauli_gadget import PPhase
        >>> from pauliopt.topologies import Topology
        >>> from pauliopt.utils import AngleExpr
        >>> from pauliopt.utils import pi
        >>> angle = AngleExpr(pi/2)
        >>> pauli_gadget = PPhase(angle) @ [X, Y, Z]
    """
    _angle: Union[AngleExpr, float]

    def __init__(self, angle: Union[AngleExpr, float]):
        self._angle = angle

    def __matmul__(self, paulis: List[Pauli]):
        return PauliGadget(self._angle, paulis)


class PauliGadget:
    """
    Class to create a PauliGadget with a given angle and paulis:
    Please note, that the angle can either be a float or an AngleExpr.
    Also note, that there exists a PPhase class to create a
    PauliGadget with a given angle:

    >>> from pauliopt.pauli.utils import Pauli, X, Y, Z, I
    >>> from pauliopt.pauli.pauli_gadget import PPhase
    >>> from pauliopt.topologies import Topology
    >>> from pauliopt.utils import AngleExpr
    >>> from pauliopt.utils import pi
    >>> angle = AngleExpr(pi/2)
    >>> pauli_gadget = PPhase(angle) @ [X, Y, Z]

    """

    def __init__(self, angle: Union[AngleExpr, float], paulis: List[Pauli]):
        self.angle = angle
        self.paulis = paulis

    def __len__(self):
        return len(self.paulis)

    def __repr__(self):
        return f"({self.angle}) @ {{ {', '.join([pauli.value for pauli in self.paulis])} }}"

    def copy(self):
        """
        Returns a deep copy of the PauliGadget
        """
        return PauliGadget(self.angle, self.paulis.copy())

    def two_qubit_count(self, topology, leg_cache=None):
        """
        Returns the amount of two qubit gates needed to implement the PauliGadget
        on a given topology.

        :param topology: Topology
        :param leg_cache: Cache for the amount of two qubit gates needed for a given column
        """
        if leg_cache is None:
            leg_cache = {}

        column = np.asarray(self.paulis)
        col_binary = np.where(column == Pauli.I, 0, 1)
        col_id = "".join([str(int(el)) for el in col_binary])
        if col_id in leg_cache.keys():
            return leg_cache[col_id]
        else:
            cnot_amount = 2 * len(find_minimal_cx_assignment(col_binary, topology)[0])
            leg_cache[col_id] = cnot_amount
        return cnot_amount

    def to_qiskit(self, topology=None):
        """
        Returns a qiskit QuantumCircuit that implements the PauliGadget on a given topology.
        If no topology is given, a complete topology is assumed.

        :param topology: Topology (default: complete)
        """

        if isinstance(self.angle, float):
            angle = self.angle
        elif isinstance(self.angle, AngleExpr):
            angle = self.angle.to_qiskit
        else:
            raise TypeError(
                f"Angle must either be float or AngleExpr, but got {type(self.angle)}")
        num_qubits = len(self.paulis)
        if topology is None:
            topology = Topology.complete(num_qubits)
        try:
            from qiskit import QuantumCircuit
        except:
            raise Exception("Please install qiskit to export Clifford Regions")
        circ = QuantumCircuit(num_qubits)

        column = np.asarray(self.paulis)
        column_binary = np.where(column == I, 0, 1)
        if np.all(column_binary == 0):
            circ.global_phase += angle
            return circ

        cnot_ladder, q0 = find_minimal_cx_assignment(column_binary, topology)
        for pauli_idx in range(len(column)):
            if column[pauli_idx] == I:
                pass
            elif column[pauli_idx] == X:
                circ.h(pauli_idx)  # Had
            elif column[pauli_idx] == Y:
                circ.rx(0.5 * np.pi, pauli_idx)  # V = Rx(0.5)
            elif column[pauli_idx] == Z:  # Z
                pass
            else:
                raise Exception(f"unknown column type: {column[pauli_idx]}")

        if len(cnot_ladder) > 0:
            for (pauli_idx, target) in reversed(cnot_ladder):
                circ.cx(pauli_idx, target)

            circ.rz(angle, q0)

            for (pauli_idx, target) in cnot_ladder:
                circ.cx(pauli_idx, target)
        else:
            target = np.argmax(column_binary)
            circ.rz(angle, target)

        for pauli_idx in range(len(column)):
            if column[pauli_idx] == Pauli.I:
                pass
            elif column[pauli_idx] == Pauli.X:
                circ.h(pauli_idx)  # Had
            elif column[pauli_idx] == Pauli.Y:
                circ.rx(-0.5 * np.pi, pauli_idx)  # Vdg = Rx(-0.5)
            elif column[pauli_idx] == Pauli.Z:
                pass
            else:
                raise Exception(f"unknown column type: {column[pauli_idx]}")
        return circ
