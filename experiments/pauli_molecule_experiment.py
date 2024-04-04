from copy import deepcopy
import os
import pickle
import shutil
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytket
import qiskit.quantum_info
import seaborn as sns
import sympy
import logging
import json
import pennylane as qml
from datetime import datetime
from pytket._tket.architecture import Architecture
from pytket._tket.passes import SequencePass, PlacementPass, RoutingPass
from pytket._tket.placement import GraphPlacement
from pytket._tket.predicates import CompilationUnit
from pytket._tket.transform import Transform, PauliSynthStrat
from pytket._tket.circuit import CXConfigType
from pytket.extensions.qiskit import tk_to_qiskit, qiskit_to_tk
from pytket.extensions.qiskit.backends import aer
from pytket.utils import gen_term_sequence_circuit, QubitPauliOperator
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli as QPauli
from qiskit.synthesis.evolution import EvolutionSynthesis, LieTrotter, QDrift
# from qiskit.providers.fake_provider import FakeGuadalupe
from qiskit.quantum_info import process_fidelity
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from sympy.core.symbol import Symbol

from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import *
from pauliopt.pauli.synthesis import PauliSynthesizer, SynthMethod
from pauliopt.utils import pi, AngleVar


def get_2q_depth(qc: QuantumCircuit):
    q = qiskit_to_tk(qc)
    return q.depth_2q()


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f'{name}.log')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(name)s: %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def generate_random_z_polynomial(num_qubits: int, num_gadgets: int, min_legs=None,
                                 max_legs=None, allowed_angels=None):
    if min_legs is None:
        min_legs = 2
    if max_legs is None:
        max_legs = num_qubits
    if allowed_angels is None:
        allowed_angels = [2 * pi, pi, pi / 2, pi /
                          4, pi / 8, pi / 16, pi / 32, pi / 64]
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
    legs = np.random.choice(
        [i for i in range(num_qubits)], size=nr_legs, replace=False)
    phase_gadget = [I for _ in range(num_qubits)]
    for leg in legs:
        phase_gadget[leg] = np.random.choice(allowed_legs)
    return PPhase(angle) @ phase_gadget


def create_random_pauli_polynomial(num_qubits: int, num_gadgets: int, min_legs=None,
                                   max_legs=None, allowed_angels=None):
    if min_legs is None:
        min_legs = 1
    if max_legs is None:
        max_legs = num_qubits
    if allowed_angels is None:
        allowed_angels = [pi, pi / 2, pi / 4, pi / 8, pi / 16]

    pp = PauliPolynomial(num_qubits)
    for _ in range(num_gadgets):
        pp >>= create_random_phase_gadget(
            num_qubits, min_legs, max_legs, allowed_angels)

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
        elif isinstance(pp[l].angle, Number):
            temp_op["coefficient"] = [float(pp[l].angle), 0]
        else:
            temp_op["coefficient"] = [float(pp[l].angle.to_qiskit), 0]
        operator_list.append(temp_op)

    temp_op_phase = {}
    p_string = []
    temp_op_phase["string"] = p_string
    if isinstance(pp.global_phase, AngleVar):
        temp_op_phase["coefficient"] = [pp.global_phase.repr_latex, 0]
    elif isinstance(pp.global_phase, Number):
        temp_op_phase["coefficient"] = [float(pp.global_phase), 0]
    else:
        temp_op_phase["coefficient"] = [float(pp.global_phase.to_qiskit), 0]
    operator_list.append(temp_op_phase)
    qubit_operator = QubitPauliOperator.from_list(operator_list)
    return qubit_operator


def pp_to_evo_operator(pp: PauliPolynomial):
    qubit_operator = QuantumCircuit(pp.num_qubits)
    for l in range(pp.num_gadgets):
        op = ''
        for q in range(pp.num_qubits):
            z_array = np.zeros(pp.num_qubits)
            x_array = np.zeros(pp.num_qubits)
            if pp[l][q] == X:
                op += 'X'
            elif pp[l][q] == Y:
                op += 'Y'
            elif pp[l][q] == Z:
                op += 'Z'
            elif pp[l][q] == I:
                op += 'I'
            else:
                raise ValueError("Pauli not recognized")

        if isinstance(pp[l].angle, AngleVar):
            print(pp[l].angle.repr_latex)
            coeff = 1
            print("AngleVar")
        elif isinstance(pp[l].angle, Number):
            coeff = float(pp[l].angle)
            # print("Number")
        else:
            coeff = float(pp[l].angle.to_qiskit)
            # print("Last")
        # print(coeff)
        pauli = QPauli(op)
        pauli_evo = PauliEvolutionGate(pauli, time=coeff, synthesis=QDrift)
        qdrift = QDrift()
        pauli_evo = qdrift.synthesize(pauli_evo)
        qubit_operator.append(pauli_evo, range(pp.num_qubits))
    return qubit_operator


def operator_to_summed_pauli_op(operator, n_qubits):
    from qiskit.opflow import I, X, Y, Z
    qps_list = list(operator._dict.keys())
    sum_pauli_op = 0.0
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
            pauli_op = paulis[0]
            for pauli in paulis[1:]:
                pauli_op = pauli_op ^ pauli

            sum_pauli_op += float(coeff) * pauli_op
        # else:
        #    sum_pauli_op += float(coeff) * I ^ n_qubits
    return sum_pauli_op


def summed_pauli_to_operator(summed_pauli, n_qubits, t=1):
    operator_list = []
    for el in summed_pauli:
        coeff = el.primitive.coeffs[0]
        pauli = el.primitive.paulis[0]
        temp_op = {}
        temp_op["string"] = [[['q', [q]], str(p)] for q, p in enumerate(pauli)
                             if str(p) != 'I']
        temp_op["coefficient"] = [coeff.real * t, coeff.imag * t]
        operator_list.append(temp_op)
    qubit_operator = QubitPauliOperator.from_list(operator_list)
    return qubit_operator


def pauli_to_pauli(p):
    if str(p) == "X":
        return X
    elif str(p) == "Y":
        return Y
    elif str(p) == "Z":
        return Z
    elif str(p) == "I":
        return I
    else:
        raise ValueError("Pauli not recognized")


def summed_pauli_to_pp(summed_pauli, n_qubits, t=1):
    pp = PauliPolynomial(n_qubits)
    for el in summed_pauli:
        coeff = float(el.primitive.coeffs[0])
        pauli = [pauli_to_pauli(p) for p in el.primitive.paulis[0]]
        pp >>= PPhase(coeff * t) @ pauli
    return pp


def operator_to_pp(operator, n_qubits, t=1):
    qps_list = list(operator._dict.keys())
    pp = PauliPolynomial(n_qubits)
    for qps in qps_list:
        coeff = operator[qps] * t
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
                pp >>= PPhase(coeff) @ paulis
            elif isinstance(coeff, sympy.core.numbers.Float):
                pp >>= PPhase(float(coeff)) @ paulis
            elif isinstance(coeff, complex):
                pp >>= PPhase(coeff) @ paulis
            elif isinstance(coeff, Number):
                pp >>= PPhase(Angle(float(coeff))) @ paulis
            elif isinstance(coeff, Symbol):
                pp >>= PPhase(AngleVar(coeff.name)) @ paulis
            else:
                raise Exception("Unknown type")
        else:
            pp.global_phase += coeff
        # idx += 1
    return pp


def synth_tket(operator, topo: Topology, method: PauliSynthStrat, n_qubits: int = None):
    if n_qubits is None:
        n_qubits = topo.num_qubits
    initial_circ = pytket.Circuit(n_qubits)
    circuit = gen_term_sequence_circuit(operator, initial_circ)

    Transform.UCCSynthesis(method, CXConfigType.Tree).apply(circuit)
    tket_arch = Architecture([(e1, e2) for (e1, e2) in topo.to_nx.edges()])
    unit = CompilationUnit(circuit)
    passes = SequencePass([
        PlacementPass(GraphPlacement(tket_arch)),
        RoutingPass(tket_arch),
    ])
    passes.apply(unit)
    circ_out = unit.circuit
    Transform.RebaseToCliffordSingles().apply(circ_out)
    Transform.RebaseToRzRx().apply(circ_out)
    return tk_to_qiskit(circ_out)


def term_sequence_tket(operator, n_qubits):
    initial_circ = pytket.Circuit(n_qubits)
    circuit = gen_term_sequence_circuit(operator, initial_circ)
    circ_out = circuit
    Transform.DecomposeBoxes().apply(circ_out)
    Transform.RebaseToCliffordSingles().apply(circ_out)
    Transform.RebaseToRzRx().apply(circ_out)
    return tk_to_qiskit(circ_out)


def synth_pp_qiskit_paulihedral(pp: PauliPolynomial, topo: Topology, prefix="qiskit_paulihedral"):
    operator = pp_to_evo_operator(pp)
    circ_out = operator
    return get_ops_count(circ_out)


def synth_pp_tket_uccs_set(pp: PauliPolynomial, topo: Topology, prefix="tket_uccs_set"):
    operator = pp_to_operator(pp)
    circ_out = synth_tket(operator, topo, PauliSynthStrat.Sets)
    return get_ops_count(circ_out)


def synth_pp_tket_uccs_pair(pp: PauliPolynomial, topo: Topology, prefix="tket_uccs_pair"):
    operator = pp_to_operator(pp)
    circ_out = synth_tket(operator, topo, PauliSynthStrat.Pairwise)
    return get_ops_count(circ_out)


def synth_pp_pauliopt_ucc(pp: PauliPolynomial, topo: Topology, prefix="pauliopt_ucc"):
    pp = simplify_pauli_polynomial(pp, allow_acs=True)
    synthesizer = PauliSynthesizer(pp, SynthMethod.UCCDS, topo)
    synthesizer.synthesize()
    return get_ops_count(synthesizer.circ_out_qiskit)


def synth_pp_pauliopt_steiner_nc(pp: PauliPolynomial, topo: Topology,
                                 prefix="pauliopt_steiner_nc"):
    pp = simplify_pauli_polynomial(pp, allow_acs=True)
    synthesizer = PauliSynthesizer(pp, SynthMethod.STEINER_GRAY_NC, topo)
    synthesizer.synthesize()
    return get_ops_count(synthesizer.circ_out_qiskit)


def synth_pp_pauliopt_steiner_clifford(pp: PauliPolynomial, topo: Topology,
                                       prefix="pauliopt_steiner_clifford"):
    pp = simplify_pauli_polynomial(pp, allow_acs=True)
    synthesizer = PauliSynthesizer(pp, SynthMethod.STEINER_GRAY_CLIFFORD, topo)
    synthesizer.synthesize()
    return get_ops_count(synthesizer.circ_out_qiskit)


def synth_pp_pauliopt_divide_conquer(pp: PauliPolynomial, topo: Topology,
                                     prefix="pauliopt_divide_conquer"):
    synthesizer = PauliSynthesizer(pp, SynthMethod.DIVIDE_AND_CONQUER, topo)
    synthesizer.synthesize()
    return get_ops_count(synthesizer.circ_out_qiskit)


def synth_pp_pauliopt_divide_conquer(pp: PauliPolynomial, topo: Topology,
                                     prefix="pauliopt_divide_conquer"):
    synthesizer = PauliSynthesizer(pp, SynthMethod.DIVIDE_AND_CONQUER, topo)
    synthesizer.synthesize()
    return get_ops_count(synthesizer.circ_out_qiskit)


def synth_pp_naive(pp: PauliPolynomial, topo: Topology, prefix="naive"):
    print("Running synth method: naive")
    circ_out = pp.to_circuit(topo).to_qiskit()
    return get_ops_count(circ_out)


def get_ops_count(qc: QuantumCircuit):
    count = {"cx": 0, "depth": 0}
    ops = qc.count_ops()
    if "cx" in ops.keys():
        count["cx"] += ops['cx']
    if "swap" in ops.keys():
        count["cx"] += ops['swap'] * 3
    count["depth"] = qc.depth()
    count["2q_depth"] = get_2q_depth(qc)
    return count


BASE_PATH = "tket_benchmarking/compilation_strategy"


def find_square_dimensions(n):
    s = int(math.sqrt(n))

    if s * s == n:
        l = k = s
        return l, k

    lower_n = n - 1
    upper_n = n + 1

    while True:
        s = int(math.sqrt(lower_n))
        if s * s == lower_n:
            l = k = s
            return l, k

        s = int(math.sqrt(upper_n))
        if s * s == upper_n:
            l = k = s
            return l, k

        lower_n -= 1
        upper_n += 1


def get_backend_and_df_name(backend_name, df_name="data/random"):
    if backend_name == "vigo":
        backend = FakeJSONBackend("ibm_vigo")
        df_name = f"{df_name}_vigo.csv"
    elif backend_name == "mumbai":
        backend = FakeJSONBackend("ibmq_mumbai")
        df_name = f"{df_name}_mumbai.csv"
    elif backend_name == "guadalupe":
        backend = FakeJSONBackend("ibmq_guadalupe")
        df_name = f"{df_name}_guadalupe.csv"
    elif backend_name == "quito":
        backend = FakeJSONBackend("ibmq_quito")
        df_name = f"{df_name}_quito.csv"
    elif backend_name == "nairobi":
        backend = FakeJSONBackend("ibm_nairobi")
        df_name = f"{df_name}_nairobi.csv"
    elif backend_name == "ithaca":
        backend = FakeJSONBackend("ibm_ithaca")
        df_name = f"{df_name}_ithaca.csv"
    elif backend_name == "seattle":
        backend = FakeJSONBackend("ibm_seattle")
        df_name = f"{df_name}_seattle.csv"
    elif backend_name == "brisbane":
        backend = FakeJSONBackend("ibm_brisbane")
        df_name = f"{df_name}_brisbane.csv"
    elif "complete" in backend_name:
        backend = "complete"
        df_name = f"{df_name}_{backend_name}.csv"
    elif "line" in backend_name:
        backend = "line"
        df_name = f"{df_name}_{backend_name}.csv"

    else:
        raise ValueError(f"Unknown backend: {backend_name}")
    return backend, df_name


def get_topo_kind(topo_kind, num_qubits):
    if topo_kind == "line":
        return Topology.line(num_qubits)
    elif topo_kind == "complete":
        return Topology.complete(num_qubits)
    elif topo_kind == "cycle":
        return Topology.cycle(num_qubits)
    elif topo_kind == "grid":
        if num_qubits == 6:
            return Topology.grid(2, 3)
        elif num_qubits == 8:
            return Topology.grid(2, 4)
        else:
            n_rows, n_cols = find_square_dimensions(num_qubits)
            return Topology.grid(n_rows, n_cols)
    else:
        raise Exception("Unknown topology kind")


SYNTHESIS_METHODS = {"tket_uccs_set": synth_pp_tket_uccs_set,
                     "tket_uccs_pair": synth_pp_tket_uccs_pair,
                     "pauliopt_steiner_nc": synth_pp_pauliopt_steiner_nc,
                     "pauliopt_ucc": synth_pp_pauliopt_ucc,
                     "pauliopt_divide_conquer": synth_pp_pauliopt_divide_conquer,
                     "pauliopt_steiner_clifford": synth_pp_pauliopt_steiner_clifford,
                     "naive": synth_pp_naive}


def create_csv_header():
    header = ["name", "num_qubits", "n_gadgets", "method",
              "cx", "depth", "2q_depth", "time"]
    return header


def synth_ucc_evaluation():
    print("Start!")
    with open(f"{BASE_PATH}/orbital_lut.txt") as json_file:
        orbitals_lookup_table = json.load(json_file)
    logger = get_logger("synth_ucc_evaluation")
    for topo_kind in ["line", "complete", "cycle"]:
        for encoding_name in ["P", "BK", "JW"]:
            op_directory = f"./datasets/pp_molecules/"
            results_file = f"data/pauli/uccsd/{topo_kind}/{encoding_name}_results.csv"
            df = pd.DataFrame({c: [] for c in create_csv_header()})
            with open(results_file, "wb") as f:
                df.to_csv(f, header=create_csv_header(), index=False)
            for filename in os.listdir(op_directory):
                name = filename.replace(".pickle", "")
                logger.info(name)
                active_spin_orbitals = orbitals_lookup_table[name]
                if encoding_name == "P":
                    n_qubits = active_spin_orbitals - 2
                else:
                    n_qubits = active_spin_orbitals
                logger.info(n_qubits)
                if n_qubits >= 15:
                    continue
                path = op_directory + "/" + filename
                with open(path, "rb") as pickle_in:
                    pp = pickle.load(pickle_in)

                topo = get_topo_kind(topo_kind, n_qubits)
                for synth_name, synth_method in SYNTHESIS_METHODS.items():

                    start = datetime.now()
                    count_dict = synth_method(pp, topo)
                    time_passed = (datetime.now()-start).total_seconds()
                    column = {"name": name,
                              "num_qubits": n_qubits,
                              "n_gadgets": pp.num_gadgets,
                              "method": synth_name,
                              "time": time_passed
                              } | count_dict

                    df = pd.DataFrame([{c: column[c]
                                      for c in create_csv_header()}])
                    with open(results_file, "ab") as f_ptr:
                        df.to_csv(f_ptr, header=False, index=False)
            print("====")


def fidelity_experiment_trotterisation():
    with open(f"{BASE_PATH}/orbital_lut.txt") as json_file:
        orbitals_lookup_table = json.load(json_file)

    logger = get_logger("fidelity_experiment_trotterisation")
    for name, encoding in [("H2_P_631g", "P"),
                           ("H4_P_sto3g", "P"),
                           ("LiH_P_sto3g", "P"),
                           ("LiH_JW_sto3g", "JW")]:

        fid_default = []
        fid_ours = []
        fid_tket = []
        logger.info(f"Name: {name}")
        for t in np.linspace(0.0, 1.0, 40):
            t = float(t * 2 * np.pi)
            logger.info(f"Time: {t}")
            # t = 1.0
            orbigtals = orbitals_lookup_table[name]
            if encoding == "P":
                n_qubits = orbigtals - 2
            else:
                n_qubits = orbigtals

            with open(
                    f"tket_benchmarking/compilation_strategy/operators/{encoding}_operators/{name}.pickle",
                    "rb") as pickle_in:
                operator = pickle.load(pickle_in)

            topo = Topology.complete(n_qubits)
            summed_pauli_operator = operator_to_summed_pauli_op(
                operator, n_qubits)

            pp = summed_pauli_to_pp(summed_pauli_operator, n_qubits, t)

            evolution_op = (
                t / 2.0 * summed_pauli_operator).exp_i().to_matrix()

            # qiskit.quantum_info.Operator(evolution_op)
            U_expected = evolution_op
            synthesizer = PauliSynthesizer(
                pp, SynthMethod.STEINER_GRAY_NC, topo)
            synthesizer.synthesize()
            U_ours = synthesizer.get_operator()
            steiner_fid = process_fidelity(U_ours, target=U_expected)
            logger.info(f"Steiner-NC fidelity: {steiner_fid}")
            fid_ours.append(steiner_fid)

            circ_tket = term_sequence_tket(operator * (t / np.pi), n_qubits)
            unitarysimulator = aer.Aer.get_backend("unitary_simulator")
            result = qiskit.execute(circ_tket, unitarysimulator).result()
            U_tket = result.get_unitary(circ_tket)
            tket_fid = process_fidelity(U_tket, target=U_expected)
            logger.info(f"tket fidelity: {tket_fid}")
            fid_tket.append(tket_fid)

            pp = summed_pauli_to_pp(summed_pauli_operator, n_qubits, t)
            unitarysimulator = aer.Aer.get_backend("unitary_simulator")
            circ = pp.to_qiskit()
            result = qiskit.execute(circ, unitarysimulator).result()
            U_default = result.get_unitary(circ)
            default_fid = process_fidelity(U_default, target=U_expected)
            logger.info(f"Default fidelity: {default_fid}")
            fid_default.append(default_fid)
        # store data
        df = pd.DataFrame({"t": np.linspace(0.0, 1.0, 40),
                           "fid_default": fid_default,
                           "fid_ours": fid_ours,
                           "fid_tket": fid_tket})
        df.to_csv(f"data/pauli/fidelity/{name}.csv")


def get_min_max_interaction(summed_pauli):
    coeff_list = []
    for el in summed_pauli:
        coeff = el.primitive.coeffs[0]
        coeff_list.append(coeff.real)
    return np.max(np.abs(coeff_list))


def plot_fidelity():
    with open(f"{BASE_PATH}/orbital_lut.txt") as json_file:
        orbitals_lookup_table = json.load(json_file)
    eps = 0.1
    for name, encoding in [("H2_P_631g", "P"),
                           ("H4_P_sto3g", "P"),
                           ("LiH_P_sto3g", "P")]:

        # t = 1.0
        orbigtals = orbitals_lookup_table[name]
        if encoding == "P":
            n_qubits = orbigtals - 2
        else:
            n_qubits = orbigtals

        with open(
                f"tket_benchmarking/compilation_strategy/operators/{encoding}_operators/{name}.pickle",
                "rb") as pickle_in:
            operator = pickle.load(pickle_in)

        summed_pauli_operator = operator_to_summed_pauli_op(operator, n_qubits)
        print("Name: ", name)
        max = get_min_max_interaction(summed_pauli_operator)
        print("Max: ", max)

        df = pd.read_csv(f"data/pauli/fidelity/{name}.csv")
        df["t"] = df["t"] * 2 * np.pi

        x_max = [t for t in df["t"] if max * t > eps][0]

        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.size": 11
        })
        sns.set_palette(sns.color_palette("colorblind"))

        sns.lineplot(df, x="t", y="fid_default", label="Default")
        sns.lineplot(df, x="t", y="fid_ours", label="PSGS")
        sns.lineplot(df, x="t", y="fid_ours", label="PSGS-PRC")
        sns.lineplot(df, x="t", y="fid_tket", label="UCCDS-set")
        # set ticks to be between 0 and 2pi and label them
        plt.xticks(np.linspace(0.0, 2 * np.pi, 5),
                   [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
        plt.xlabel(r"Time $t$")
        plt.ylabel(r"Fidelity")
        # add  vertical line at t = 1
        plt.axvline(x=x_max, color=sns.color_palette(
            "colorblind")[4], linestyle="--")
        # add a label to the vertical line
        plt.text(x=x_max + 0.1, y=0.01, s=r"$\vec{t}_{max} > 0.1$")

        plt.tight_layout()
        plt.savefig(f"data/pauli/fidelity/{name}.pdf")
        plt.show()


def bold_max(df, alg, type, row):
    values = [f'tket_uccs_set_{type}',
              f'pauliopt_steiner_nc_{type}',
              f"pauliopt_ucc_{type}"]
    max_val = min(df.loc[row.name, values])
    curr_alg = f"{alg}{type}"
    outstring = f"{row[f'{alg}{type}']} ({row[f'{alg}{type}_reduction']:.2f})"
    if row[curr_alg] == max_val:
        return f"\\textbf{{{outstring}}}"
    else:
        return outstring


def sanitize_uccsd_molecules():
    for arch_name in ["complete", "line", "cycle"]:
        if not os.path.exists(f"data/pauli/uccsd_san/{arch_name}"):
            os.makedirs(f"data/pauli/uccsd_san/{arch_name}")
        else:
            shutil.rmtree(f"data/pauli/uccsd_san/{arch_name}")
            os.makedirs(f"data/pauli/uccsd_san/{arch_name}")

        for encoding in ["BK", "JW", "P"]:
            df = pd.read_csv(
                f"data/pauli/uccsd/{arch_name}/{encoding}_results.csv")
            del df["Unnamed: 0"]
            del df["tket_uccs_pair_cx"]
            del df["tket_uccs_pair_depth"]
            df.sort_values(by=["n_qubits", "gadgets"], inplace=True)
            df.reset_index(drop=True, inplace=True)

            for alg, rename in [("tket_uccs_set_", "UCCSD-set"),
                                ("pauliopt_steiner_nc_",
                                 "pauli-steiner-gray-synth"),
                                ("pauliopt_ucc_", "architecture-aware-UCCSD-set")]:
                for type in ["cx", "depth"]:
                    df[f"{alg}{type}_reduction"] \
                        = (df[f"naive_{type}"] - df[f"{alg}{type}"]) / df[
                        f"naive_{type}"] * 100

                    # assign df, alg, type to lambda function
                    def bold_max_(row): return bold_max(df, alg, type, row)

                    # create a new column {rename} ({type}) which is formatted as a string: "{{alg}{type}}, ({alg}{type}_reduction)"
                    df[f"{rename} ({type})"] = df.apply(bold_max_, axis=1)
                    # drop the old columns

            for alg, rename in [("tket_uccs_set_", "UCCSD-set"),
                                ("pauliopt_steiner_nc_",
                                 "pauli-steiner-gray-synth"),
                                ("pauliopt_steiner_clifford",
                                 "pauli-steiner-gray-synth-clifford"),
                                ("pauliopt_ucc_", "architecture-aware-UCCSD-set")]:
                for type in ["cx", "depth"]:
                    del df[f"{alg}{type}"]
                    del df[f"{alg}{type}_reduction"]

            del df["naive_cx"]
            del df["naive_depth"]

            df.to_csv(
                f"data/pauli/uccsd_san/{arch_name}/{encoding}_results.csv")
            with open(f"data/pauli/uccsd_san/{arch_name}/{encoding}_results.tex",
                      "w") as f:
                df.to_latex(f, index=False, escape=False, float_format="%.2f")


if __name__ == '__main__':
    synth_ucc_evaluation()
