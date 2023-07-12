import pickle
import time
import warnings

import numpy as np
import pytket
import stim
from pytket._tket.architecture import Architecture
from pytket._tket.partition import term_sequence, PauliPartitionStrat, GraphColourMethod
from pytket._tket.passes import SequencePass, PlacementPass, RoutingPass
from pytket._tket.placement import GraphPlacement
from pytket._tket.predicates import CompilationUnit
from pytket._tket.transform import Transform, PauliSynthStrat, CXConfigType
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.utils import gen_term_sequence_circuit
from qiskit import QuantumCircuit

from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import *
from pauliopt.pauli.synth.uccds import uccds
from pauliopt.pauli.synth.divide_and_conquer import divide_and_conquer
from pauliopt.pauli.synth.steiner_gray_nc import pauli_polynomial_steiner_gray_nc
from pauliopt.pauli.synthesis import PauliSynthesizer, SynthMethod
from pauliopt.pauli.utils import apply_permutation
from pauliopt.utils import pi


def generate_random_z_polynomial(num_qubits: int, num_gadgets: int, min_legs=None,
                                 max_legs=None, allowed_angels=None):
    if min_legs is None:
        min_legs = 2
    if max_legs is None:
        max_legs = num_qubits
    if allowed_angels is None:
        allowed_angels = [2 * pi, pi, pi / 2, pi / 4, pi / 8, pi / 16, pi / 32, pi / 64]
    allowed_legs = [Z]
    pp = PauliPolynomial(num_qubits)
    for _ in range(num_gadgets):
        pp >>= create_random_phase_gadget(num_qubits, min_legs, max_legs,
                                          allowed_angels, allowed_legs=allowed_legs)
    return pp


def create_random_phase_gadget(num_qubits, min_legs, max_legs, allowed_angels,
                               allowed_legs=None):
    if allowed_legs is None:
        allowed_legs = [X, Y, Z]
    angle = np.random.choice(allowed_angels)
    nr_legs = np.random.randint(min_legs, max_legs)
    legs = np.random.choice([i for i in range(num_qubits)], size=nr_legs, replace=False)
    phase_gadget = [I for _ in range(num_qubits)]
    for leg in legs:
        phase_gadget[leg] = np.random.choice(allowed_legs)
    return PPhase(angle) @ phase_gadget


def generate_random_pauli_polynomial(num_qubits: int, num_gadgets: int, min_legs=None,
                                     max_legs=None, allowed_angels=None):
    if min_legs is None:
        min_legs = 1
    if max_legs is None:
        max_legs = num_qubits
    if allowed_angels is None:
        allowed_angels = [pi, pi / 2, pi / 4, pi / 8, pi / 16]

    pp = PauliPolynomial(num_qubits)
    for _ in range(num_gadgets):
        pp >>= create_random_phase_gadget(num_qubits, min_legs, max_legs, allowed_angels)

    return pp


def operator_to_pp(operator, n_qubits,
                   partition_strat=PauliPartitionStrat.CommutingSets,
                   colour_method=GraphColourMethod.Lazy, ):
    qps_list = list(operator._dict.keys())
    qps_list_list = term_sequence(qps_list, partition_strat, colour_method)
    pp = PauliPolynomial(n_qubits)
    for out_qps_list in qps_list_list:
        for qps in out_qps_list:
            coeff = operator[qps]
            qps_map = qps.map
            if qps_map:
                paulis = [I for _ in range(n_qubits)]
                for qb, pauli in qps_map.items():
                    if pauli == 0:
                        continue
                    elif pauli == 1:
                        paulis[qb.index[0]] = X
                    elif pauli == 2:
                        paulis[qb.index[0]] = Y
                    elif pauli == 3:
                        paulis[qb.index[0]] = Z
                pp >>= PPhase(pi / 4) @ paulis
            # idx += 1
    return pp


def synth_pp_tket_uccds(pp: PauliPolynomial, topo: Topology):
    pass
