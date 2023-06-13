from enum import Enum

from pauliopt.utils import Angle

COMMUTATION_RULES = {('X', 'X'): ('I', 1),
                     ('X', 'Y'): ('Z', 1j),
                     ('X', 'Z'): ('Y', -1j),
                     ('X', 'I'): ('X', 1),
                     ('Y', 'X'): ('Z', -1j),
                     ('Y', 'Y'): ('I', 1),
                     ('Y', 'Z'): ('X', 1j),
                     ('Y', 'I'): ('Y', 1),
                     ('Z', 'X'): ('Y', 1j),
                     ('Z', 'Y'): ('X', -1j),
                     ('Z', 'Z'): ('I', 1),
                     ('Z', 'I'): ('Z', 1),
                     ('I', 'X'): ('X', 1),
                     ('I', 'Y'): ('Y', 1),
                     ('I', 'Z'): ('Z', 1),
                     ('I', 'I'): ('I', 1)}


class Pauli(Enum):
    I = "I"
    X = "X"
    Y = "Y"
    Z = "Z"

    def __add__(self, other):
        if isinstance(other, Pauli):
            weights = [1, 1]
            pauli_strings = [self, other]
            return Hamiltonian(pauli_strings, weights)
        elif isinstance(other, Hamiltonian):
            pass
        else:
            raise NotImplementedError("Pauli can only be added to Pauli or Hamiltonian")

    def __mul__(self, other):
        if isinstance(other, Pauli):
            res_pauli, res_phase = COMMUTATION_RULES[(self.value, other.value)]
            return Pauli(res_pauli) * Angle(res_phase)
        elif isinstance(other, Hamiltonian):
            if other.size() != 2:
                raise Exception("Hamiltonian must be of size 2")
            return other * self
        elif isinstance(other, Angle):
            return Hamiltonian([self], [other])
        else:
            raise NotImplementedError(
                "Pauli can only be multiplied by Pauli or Hamiltonian")

    def __pow__(self, power, modulo=None):
        pass


class Hamiltonian:
    """
    We define a hamiltonian as a sum of weighted pauli strings.
    """

    def __init__(self, pauli_strings, weights):
        len_0 = len(pauli_strings[0])
        for pauli in pauli_strings:
            if len(pauli) != len_0:
                raise Exception("Pauli strings must be of equal length")
        if not len(pauli_strings) == len(weights):
            raise Exception("Pauli strings and weights must be of equal length")
        self.pauli_strings = pauli_strings
        self.weights = weights

    def size(self):
        return 2 ** len(self.pauli_strings[0])

    def __len__(self):
        return len(self.pauli_strings)

    def __add__(self, other):
        if isinstance(other, Pauli):
            if not self.size() == 2:
                raise Exception(
                    "A single Pauli can only be added to Hamiltonian of size 2")
            self.weights.append(1)
            self.pauli_strings.append(other)
            return self
        elif isinstance(other, Hamiltonian):
            return Hamiltonian(self.pauli_strings + other.pauli_strings,
                               self.weights + other.weights)
        else:
            raise NotImplementedError(
                "Hamiltonian can only be added to Pauli or Hamiltonian")

    def __mul__(self, other):
        if isinstance(other, Pauli):
            pass
        elif isinstance(other, Hamiltonian):
            pass
        elif isinstance(other, Angle):
            pass

I = Pauli.I
X = Pauli.X
Y = Pauli.Y
Z = Pauli.Z
