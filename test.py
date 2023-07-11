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


def verify_equality_unitary(qc_in, qc_out):
    """
    Verify the equality up to a global phase
    :param qc_in:
    :param qc_out:
    :return:
    """
    try:
        from qiskit.quantum_info import Operator
    except:
        raise Exception("Please install qiskit to compare to quantum circuits")
    return np.allclose(Operator.from_circuit(qc_in).data,
                       Operator.from_circuit(qc_out).data)


def verify_equality(qc_in, qc_out):
    """
    Verify the equality up to a global phase
    :param qc_in:
    :param qc_out:
    :return:
    """
    try:
        from qiskit.quantum_info import Statevector, Operator
    except:
        raise Exception("Please install qiskit to compare to quantum circuits")
    return Statevector.from_instruction(qc_in) \
        .equiv(Statevector.from_instruction(qc_out))


def random_hscx_circuit(nr_gates=20, nr_qubits=4):
    gate_choice = ["CX", "H", "S"]
    qc = QuantumCircuit(nr_qubits)
    for _ in range(nr_gates):
        gate_t = np.random.choice(gate_choice)
        if gate_t == "H":
            qubit = np.random.choice([i for i in range(nr_qubits)])
            qc.h(qubit)
        elif gate_t == "S":
            qubit = np.random.choice([i for i in range(nr_qubits)])
            qc.s(qubit)
        elif gate_t == "CX":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cx(control, target)
        elif gate_t == "CY":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cy(control, target)
        elif gate_t == "CZ":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice([i for i in range(nr_qubits) if i != control])
            qc.cz(control, target)
    return qc


def undo_permutation(qc: QuantumCircuit, perm):
    circ_out = qiskit_to_tk(qc)
    inv_map = {circ_out.qubits[v]: circ_out.qubits[k] for v, k in enumerate(perm)}
    circ_out_ = pytket.Circuit(circ_out.n_qubits)
    for cmd in circ_out:
        if isinstance(cmd, pytket.circuit.Command):
            remaped_qubits = list(map(lambda node: inv_map[node], cmd.qubits))
            circ_out_.add_gate(cmd.op, remaped_qubits)
    circ_out = tk_to_qiskit(circ_out_)
    return circ_out


def check_matching_architecture(qc: QuantumCircuit, G):
    for gate in qc:
        if gate.operation.num_qubits == 2:
            ctrl, target = gate.qubits
            ctrl, target = ctrl._index, target._index  # TODO refactor this to a non deprecated way
            if not G.has_edge(ctrl, target):
                return False
    return True


def route_circuit_tket(circuit: pytket.Circuit, topo: Topology, transform="naive"):
    if transform == "naive":
        Transform.DecomposeBoxes().apply(circuit)
    else:
        Transform.UCCSynthesis(PauliSynthStrat.Sets, CXConfigType.Tree).apply(circuit)
    tket_arch = Architecture([e for e in topo.to_nx.edges()])
    unit = CompilationUnit(circuit)
    passes = SequencePass([
        PlacementPass(GraphPlacement(tket_arch)),
        RoutingPass(tket_arch),
    ])
    passes.apply(unit)
    circ_out = unit.circuit
    Transform.RebaseToCliffordSingles().apply(circ_out)
    Transform.RebaseToRzRx().apply(circ_out)
    return circ_out


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


def permute_pp(pp: PauliPolynomial, permutation: list):
    swapped = [False] * pp.num_qubits

    for idx, i in enumerate(permutation):
        if i != idx and not swapped[i] and not swapped[idx]:
            swapped[i] = True
            swapped[idx] = True
            pp.swap_rows(idx, i)
    return pp


def test_sto3g():
    # H2_BK_sto3g
    # H2_P_631g
    with open("H2_BK_sto3g.pickle", "rb") as pickle_in:
        qubit_pauli_operator = pickle.load(pickle_in)

    n_qubits = 6
    topo = Topology.line(n_qubits)

    pp = operator_to_pp(qubit_pauli_operator, n_qubits)
    # pp = simplify_pauli_polynomial(pp, allow_acs=False)
    circ_out, gadget_parm, perm = pauli_polynomial_steiner_gray_nc(pp.copy(), topo)
    pp_ = PauliPolynomial(pp.num_qubits)
    pp_.pauli_gadgets = [pp[i].copy() for i in gadget_parm]
    assert verify_equality(circ_out, pp_.to_qiskit(topo))
    circ_out = apply_permutation(circ_out, perm)
    assert check_matching_architecture(circ_out, topo.to_nx)
    # circ_out = apply_permutation(circ_out, perm)
    # print(circ_out)
    initial_circ = pytket.Circuit(n_qubits)
    set_synth_circuit = gen_term_sequence_circuit(qubit_pauli_operator, initial_circ)
    naive_circuit = route_circuit_tket(set_synth_circuit.copy(),
                                       topo, transform="naive")
    naive_circuit = tk_to_qiskit(naive_circuit)

    uccds_circuit = route_circuit_tket(set_synth_circuit.copy(),
                                       topo, transform="other")
    uccds_circuit = tk_to_qiskit(uccds_circuit)
    print("Naive with tket:  ", naive_circuit.count_ops(), naive_circuit.depth())
    print("Synth tket uccds: ", uccds_circuit.count_ops(), uccds_circuit.depth())
    print("Ours:             ", circ_out.count_ops(), circ_out.depth())


def main():
    pp = generate_random_pauli_polynomial(4, 100)
    topo = Topology.line(pp.num_qubits)
    start = time.time()
    synthesizer = PauliSynthesizer(pp, SynthMethod.DIVIDE_AND_CONQUER, topo)
    synthesizer.synthesize()
    print("Time taken: ", time.time() - start)
    circ_out = synthesizer.circ_out

    print("Ours:   ", circ_out.count_ops()["cx"])
    print("PP:     ", pp.to_qiskit(topology=topo).count_ops()["cx"])
    print("Ours:   ", circ_out.depth())
    print("PP:     ", pp.to_qiskit(topology=topo).depth())


if __name__ == '__main__':
    main()
