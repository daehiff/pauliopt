import types
from enum import Enum
from typing import SupportsFloat, SupportsComplex

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
            pauli_strings = [[self], [other]]
            return Hamiltonian(pauli_strings, weights)
        elif isinstance(other, Hamiltonian):
            if other.size() != 2:
                raise Exception("Hamiltonian must be of size 2")
            return Hamiltonian([[self]] + other.pauli_strings,
                               [1] + other.weights)
        else:
            raise NotImplementedError("Pauli can only be added to Pauli or Hamiltonian")

    def __mul__(self, other):
        # self * other
        if isinstance(other, (SupportsFloat, SupportsComplex)):
            return Hamiltonian([self], [other])
        elif isinstance(other, Pauli):
            res_pauli, res_phase = COMMUTATION_RULES[(self.value, other.value)]
            return Hamiltonian([[Pauli(res_pauli)]], [res_phase])
        elif isinstance(other, Hamiltonian):
            if other.size() != 2:
                raise Exception("Hamiltonian must be of size 2")
            return other * self
        else:
            raise NotImplementedError(
                "Pauli can only be multiplied by Pauli, Number or Hamiltonian")

    def __rmul__(self, other):
        # other * self
        if isinstance(other, (SupportsFloat, SupportsComplex)):
            return Hamiltonian([[self]], [other])
        elif isinstance(other, Pauli):
            res_pauli, res_phase = COMMUTATION_RULES[(other.value, self.value)]
            return Hamiltonian([[Pauli(res_pauli)]], [res_phase])
        elif isinstance(other, Hamiltonian):
            if other.size() != 2:
                raise Exception("Hamiltonian must be of size 2")
            return other * self
        else:
            raise NotImplementedError(
                "Pauli can only be multiplied by Pauli or Hamiltonian")

    def __pow__(self, other, modulo=None):
        if isinstance(other, Pauli):
            return Hamiltonian([[self, other]], [1])
        elif isinstance(other, int):
            if other < 0:
                raise NotImplementedError(
                    "Pauli can only be raised to positive integers")
            elif other == 0:
                return Hamiltonian([Pauli.I], [1])
            else:
                return Hamiltonian([self] * other, [1] * other)
        elif isinstance(other, Hamiltonian):
            return Hamiltonian([[self] + p_string for p_string in other.pauli_strings],
                               other.weights)
        else:
            raise NotImplementedError(
                "Pauli can only be raised to Pauli, int or Hamiltonian")


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
            self.pauli_strings.append([other])
            return self
        elif isinstance(other, Hamiltonian):
            return Hamiltonian(self.pauli_strings + other.pauli_strings,
                               self.weights + other.weights)
        else:
            raise NotImplementedError(
                "Hamiltonian can only be added to Pauli or Hamiltonian")

    def __mul__(self, other):
        if isinstance(other, Pauli):
            if not self.size() == 2:
                raise Exception(
                    "A single Pauli can only be multiplied by Hamiltonian of size 2")
            p_strings_ = []
            for idx, p_string in enumerate(self.pauli_strings):
                p_string = p_string[0]
                res_pauli, res_phase = COMMUTATION_RULES[(p_string.value, other.value)]
                p_strings_.append([Pauli(res_pauli)])
                self.weights[idx] *= res_phase
            return Hamiltonian(p_strings_, self.weights)
        elif isinstance(other, Hamiltonian):
            if not self.size() == other.size():
                raise Exception("Hamiltonians must be of equal size")
            p_strings_ = []
            for idx, (p_string_, other_string) in enumerate(
                    zip(self.pauli_strings, other.pauli_strings)):
                new_string = []
                for p_, p_pther in zip(p_string_, other_string):
                    res_pauli, res_phase = COMMUTATION_RULES[(p_.value, p_pther.value)]
                    self.weights[idx] *= res_phase
                    new_string.append(Pauli(res_pauli))
                p_strings_.append(new_string)
            return Hamiltonian(p_strings_, self.weights)
        elif isinstance(other, (SupportsFloat, SupportsComplex)):
            self.weights = [weight * other for weight in self.weights]
            return self

    def __repr__(self):
        reps_string = ""
        for pauli_string, weight in zip(self.pauli_strings, self.weights):
            pauli_string_reps = "".join([pauli.value for pauli in pauli_string])
            reps_string += f"{weight}*{pauli_string_reps} + \n"
        return reps_string[:-4]

    def __pow__(self, other, modulo=None):
        # H ** other
        if isinstance(other, Pauli):
            return Hamiltonian(
                [[other] + pauli_string for pauli_string in self.pauli_strings],
                self.weights)
        elif isinstance(other, Hamiltonian):
            p_strings_ = []
            weigths = []
            for weight, p_string in zip(self.weights, self.pauli_strings):
                for other_weight, other_p_string in zip(other.weights,
                                                        other.pauli_strings):
                    p_new = p_string + other_p_string
                    weight = weight * other_weight
                    p_strings_.append(p_new)
                    weigths.append(weight)
            return Hamiltonian(p_strings_, weigths)
        else:
            raise NotImplementedError(
                "Hamiltonian can only be raised to Pauli or Hamiltonian")


I = Pauli.I
X = Pauli.X
Y = Pauli.Y
Z = Pauli.Z
