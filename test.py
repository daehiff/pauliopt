import pickle
import time
from numbers import Number

import numpy as np
import pandas as pd
import pytket
from matplotlib import pyplot as plt
from mqt import qmap
from pytket._tket.architecture import Architecture
from pytket._tket.passes import SequencePass, PlacementPass, RoutingPass
from pytket._tket.placement import GraphPlacement
from pytket._tket.predicates import CompilationUnit
from pytket._tket.transform import Transform, PauliSynthStrat, CXConfigType
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.utils import gen_term_sequence_circuit, QubitPauliOperator
from pyzx import Mat2
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Clifford, random_clifford
from sympy.core.symbol import Symbol

from pauliopt.pauli.clifford_tableau import CliffordTableau
from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import *
from pauliopt.pauli.synthesis import PauliSynthesizer, SynthMethod
from pauliopt.utils import pi, AngleVar, π
import seaborn as sns


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


def operator_to_pp(operator, n_qubits):
    qps_list = list(operator._dict.keys())
    pp = PauliPolynomial(n_qubits)
    for qps in qps_list:
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
            if isinstance(coeff, float):
                pp >>= PPhase(Angle(coeff)) @ paulis
            elif isinstance(coeff, complex):
                pp >>= PPhase(Angle(coeff.real)) @ paulis
            elif isinstance(coeff, Number):
                pp >>= PPhase(Angle(float(coeff))) @ paulis
            elif isinstance(coeff, Symbol):
                pp >>= PPhase(AngleVar(coeff.name)) @ paulis
            else:
                raise Exception("Unknown type")
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


def pp_to_operator(pp: PauliPolynomial):
    operator_list = []
    for l in range(pp.num_gadgets):
        temp_op = {}
        p_string = [[['q', [q]], pp[l][q].value] for q in range(pp.num_qubits)
                    if pp[l][q] != I]
        temp_op["string"] = p_string
        if isinstance(pp[l].angle, AngleVar):
            temp_op["coefficient"] = pp[l].angle.repr_latex
        else:
            temp_op["coefficient"] = [pp[l].angle.to_qiskit, 0]
        operator_list.append(temp_op)
    qubit_operator = QubitPauliOperator.from_list(operator_list)
    return qubit_operator


def test_sto3g():
    # H2_BK_sto3g
    # H2_P_631g
    with open("H2_BK_sto3g.pickle", "rb") as pickle_in:
        qubit_pauli_operator = pickle.load(pickle_in)

    n_qubits = 4
    topo = Topology.complete(n_qubits)

    pp = operator_to_pp(qubit_pauli_operator, n_qubits)
    pp = simplify_pauli_polynomial(pp, allow_acs=True)
    print(pp)
    synthesizer = PauliSynthesizer(pp, SynthMethod.STEINER_GRAY_NC, topo)
    synthesizer.synthesize()
    # assert synthesizer.check_connectivity_predicate()
    # assert synthesizer.check_circuit_equivalence()
    circ_out = synthesizer.circ_out_qiskit

    initial_circ = pytket.Circuit(n_qubits)
    set_synth_circuit = gen_term_sequence_circuit(qubit_pauli_operator, initial_circ)
    naive_circuit = route_circuit_tket(set_synth_circuit.copy(),
                                       topo, transform="naive")
    naive_circuit = tk_to_qiskit(naive_circuit)

    uccds_circuit = route_circuit_tket(set_synth_circuit.copy(),
                                       topo, transform="other")
    uccds_circuit = tk_to_qiskit(uccds_circuit)
    print(uccds_circuit)
    print(circ_out)
    print("Naive with tket:  ", naive_circuit.count_ops(), naive_circuit.depth())
    print("Synth tket uccds: ", uccds_circuit.count_ops(), uccds_circuit.depth())
    print("Ours:             ", circ_out.count_ops(), circ_out.depth())


def main():
    pp = PauliPolynomial(5)
    # pp >>= PPhase(π) @ [I, Y, I, I, I]
    pp >>= PPhase(π / 2) @ [X, I, I, Y, X]
    pp >>= PPhase(π / 2) @ [Z, Y, I, I, I]
    pp >>= PPhase(π / 8) @ [Y, Z, I, Y, Z]
    # pp >>= PPhase(π / 16) @ [I, I, I, Y, I]
    # pp >>= PPhase(π / 2) @ [X, Z, Z, X, I]
    # pp >>= PPhase(π) @ [I, Z, X, Y, Z]
    # pp >>= PPhase(π / 4) @ [X, I, X, I, Z]
    # pp >>= PPhase(π / 16) @ [X, Y, Y, Y, I]
    # pp >>= PPhase(π / 4) @ [X, X, Z, Y, I]

    # pp = generate_random_pauli_polynomial(5, 5)
    print(pp)
    topo = Topology.complete(pp.num_qubits)
    start = time.time()
    synthesizer = PauliSynthesizer(pp, SynthMethod.STEINER_GRAY_NC, topo)
    synthesizer.synthesize()
    print("Time taken: ", time.time() - start)
    circ_out = synthesizer.circ_out_qiskit
    print(circ_out)
    print("Steiner:   ", circ_out.count_ops()["cx"])
    print("PP:     ", pp.to_qiskit(topology=topo).count_ops()["cx"])
    assert (synthesizer.check_circuit_equivalence())
    # synthesizer = PauliSynthesizer(pp, SynthMethod.UCCDS, topo)
    # synthesizer.synthesize()
    # print("Time taken: ", time.time() - start)
    # circ_out = synthesizer.circ_out_qiskit
    #
    # print("UCCDS:   ", circ_out.count_ops()["cx"])
    # print("PP:     ", pp.to_qiskit(topology=topo).count_ops()["cx"])

    # synthesizer = PauliSynthesizer(pp, SynthMethod.DIVIDE_AND_CONQUER, topo)
    # synthesizer.synthesize()
    # print("Time taken: ", time.time() - start)
    # circ_out = synthesizer.circ_out_qiskit
    #
    # print("Divide:   ", circ_out.count_ops()["cx"])
    # print("PP:     ", pp.to_qiskit(topology=topo).count_ops()["cx"])
    # print("Ours:   ", circ_out.depth())
    # print("PP:     ", pp.to_qiskit(topology=topo).depth())


class CNOT_tracker:
    def __init__(self):
        self.cnots = []

    def row_add(self, target, control):
        self.cnots.append((target, control))

    def col_add(self, target, control):
        self.cnots.append((control, target))


def random_parity_map():
    mat = Mat2.id(4)
    for _ in range(10):
        control = np.random.choice(list(range(4)))
        target = np.random.choice([x for x in range(4) if x != control])
        mat.row_add(target, control)
    print(mat)
    tracker = CNOT_tracker()

    mat.copy().gauss(full_reduce=True, x=tracker)

    print(tracker.cnots)


def clifford_tableau_fun():
    matrix = np.array([
        [0, 0, 1, 0, 0, 1],
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0]
    ])
    signs = np.array([0, 0, 0, 0, 0, 0])

    ct = CliffordTableau(tableau=matrix, signs=signs)
    ct.append_s(2)
    ct.append_cnot(2, 0)
    ct.append_cnot(0, 2)
    ct.append_cnot(2, 0)
    ct.append_cnot(1, 2)
    ct.print_zx()


def pp_decomposition():
    pp = PauliPolynomial(2)

    pp >>= PPhase(AngleVar("\\alpha_0")) @ [Y, Z]
    pp >>= PPhase(AngleVar("\\alpha_1")) @ [Z, Z]

    print(pp.to_latex())


def test_clifford_1001():
    start = time.time()
    circ = random_hscx_circuit(10, 5)
    topo = Topology.line(circ.num_qubits)
    coupling = [list(x) for x in topo.couplings]

    circ_ = transpile(circ,
                      initial_layout=[x for x in range(circ.num_qubits)],
                      basis_gates=["cx", "h", "s"],
                      coupling_map=coupling)
    print("Original: ", circ.count_ops())

    ct = CliffordTableau.from_circuit(circ)
    ct.print_zx()

    circ_out_opt = ct.optimal_to_circuit(topo)

    circ_out_ours, _ = ct.to_clifford_circuit_arch_aware_qiskit(topo,
                                                                include_swaps=False)
    tableau = Clifford.from_circuit(circ)
    circ_out_bravi = tableau.to_circuit()
    circ_out_bravi = transpile(circ_out_bravi,
                               initial_layout=[x for x in range(circ.num_qubits)],
                               coupling_map=coupling,
                               basis_gates=["cx", "h", "s"])

    print(verify_equality(circ_out_opt, circ))

    print("Optimal:   ", circ_out_opt.count_ops()["cx"])
    print("Bravi:     ", circ_out_bravi.count_ops()["cx"])
    print("Ours:      ", circ_out_ours.count_ops()["cx"])
    print("Original:  ", circ_.count_ops()["cx"])
    print("dt:        ", time.time() - start)


def test_clifford_1000(num_qubits=7):
    def get_ops(circ):
        return circ.count_ops()["cx"] if "cx" in circ.count_ops() else 0

    topo = Topology.complete(num_qubits)
    coupling = [list(x) for x in topo.couplings]

    df = pd.DataFrame(columns=["output", "input", "method"])
    for gates in range(5, 20, 3):
        print(gates)
        for _ in range(5):
            circ = random_hscx_circuit(gates, num_qubits)
            circ_ = transpile(circ,
                              initial_layout=[x for x in range(circ.num_qubits)],
                              coupling_map=coupling,
                              basis_gates=["cx", "h", "s"])
            if get_ops(circ_) == 0:
                continue
            ct = CliffordTableau.from_circuit(circ)

            circ_out_opt = ct.optimal_to_circuit(topo)

            circ_out_ours, _ = ct.to_clifford_circuit_arch_aware_qiskit(topo,
                                                                        include_swaps=False)

            tableau = Clifford.from_circuit(circ)
            circ_out_bravi = tableau.to_circuit()
            circ_out_bravi = transpile(circ_out_bravi,
                                       initial_layout=[x for x in range(circ.num_qubits)],
                                       coupling_map=coupling,
                                       basis_gates=["cx", "h", "s"])

            bravi = pd.DataFrame({"output": get_ops(circ_out_bravi) / get_ops(circ_),
                                  "input": gates,
                                  "method": "Bravi"}, index=[0])
            ours = pd.DataFrame({"output": get_ops(circ_out_ours) / get_ops(circ_),
                                 "input": gates,
                                 "method": "Ours"}, index=[0])
            opt = pd.DataFrame({"output": get_ops(circ_out_opt) / get_ops(circ_),
                                "input": gates,
                                "method": "Optimal"}, index=[0])

            df = pd.concat([df, bravi, ours, opt], ignore_index=True)

    sns.lineplot(x="input", y="output", hue="method", data=df)
    plt.show()


def test_tableau_1001(n_qubits=6):
    clifford = random_clifford(n_qubits)

    circ_bravi = clifford.to_circuit()
    circ_bravi = transpile(circ_bravi, optimization_level=0, basis_gates=["cx", "h", "s"])
    # exit(0)
    ct = CliffordTableau.from_circuit(circ_bravi)
    circ_out = ct.to_clifford_circuit_opt()
    circ_out = transpile(circ_out, optimization_level=0, basis_gates=["cx", "h", "s"])
    print(circ_out.count_ops())
    print(circ_bravi.count_ops())

    print("lb: ", math.ceil(n_qubits ** 2 / math.log2(n_qubits)))
    print("Ours: ", circ_out.count_ops()["cx"])
    print("Bravi: ", circ_bravi.count_ops()["cx"])
    print("UB: ", n_qubits ** 2)
    return math.ceil(n_qubits ** 2 / math.log(n_qubits)), \
           circ_out.count_ops()["cx"], \
           circ_bravi.count_ops()["cx"], n_qubits ** 2


if __name__ == '__main__':
    qubits = list(range(3, 50, 5))
    ours_all = []
    bravi_all = []
    lb_all = []
    ub_all = []
    for n_qubits in qubits:
        lb, ours, bravi, ub = test_tableau_1001(n_qubits)
        ours_all.append(ours)
        bravi_all.append(bravi)
        lb_all.append(lb)
        ub_all.append(ub)

    plt.plot(qubits, ours_all, label="Ours")
    plt.plot(qubits, bravi_all, label="Bravi")
    plt.plot(qubits, lb_all, label="Lower bound")
    plt.plot(qubits, ub_all, label="Upper bound")
    plt.legend()
    plt.show()
    # test_tableau_1001(20)
    # test_tableau_1001()
    # pp_decomposition()
