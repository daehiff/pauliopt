import numpy as np
import scipy.linalg

from pauliopt.pauli.utils import Y, X, Z, I




def pauli_to_matrix(pauli):
    if pauli.value == "X":
        return np.asarray([[0, 1],
                           [1, 0]])
    elif pauli.value == "Y":
        return np.asarray([[0, -1j],
                           [1j, 0]])
    elif pauli.value == "Z":
        return np.asarray([[1, 0],
                           [0, -1]])
    elif pauli.value == "I":
        return np.asarray([[1, 0],
                           [0, 1]])
    else:
        raise Exception(f"Unknown Pauli: {pauli}")


def n_fold_kron(paulis):
    kron = pauli_to_matrix(paulis[0])
    for pauli in paulis[1:]:
        kron = np.kron(kron, pauli_to_matrix(pauli))
    return kron


def get_pauli_operator(pauli_strings, phases):
    sum = 0
    for pauli_string, phase in zip(pauli_strings, phases):
        sum += phase * n_fold_kron(pauli_string)
    return scipy.linalg.expm(-1j * 0.5 * sum)


def get_resulting_pauli(p_mat, p1_mat):
    for pauli in [X, Y, Z, I]:
        for phase in [1, -1, 1.j, -1.j]:
            res_mat = pauli_to_matrix(pauli)
            if np.allclose(p_mat @ p1_mat, phase * res_mat):
                return (pauli.value, phase)


def main():
    print((2*X) ** Y ** Z)


if __name__ == '__main__':
    main()
