import os
import pickle
from numbers import Number

import matplotlib.pyplot as plt
import pandas as pd
import pytket
import qiskit.quantum_info
import seaborn as sns
import sympy
from pytket._tket.architecture import Architecture
from pytket._tket.passes import SequencePass, PlacementPass, RoutingPass
from pytket._tket.placement import GraphPlacement
from pytket._tket.predicates import CompilationUnit
from pytket._tket.transform import Transform, PauliSynthStrat, CXConfigType
from pytket.extensions.qiskit import tk_to_qiskit
from pytket.extensions.qiskit.backends import aer
from pytket.utils import gen_term_sequence_circuit, QubitPauliOperator
from qiskit import QuantumCircuit
from qiskit.quantum_info import process_fidelity
from sympy.core.symbol import Symbol

from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import *
from pauliopt.pauli.synthesis import PauliSynthesizer, SynthMethod
from pauliopt.utils import pi, AngleVar
import logging


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
        pp >>= create_random_phase_gadget(num_qubits, min_legs, max_legs, allowed_angels)

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


def synth_tket(operator, topo: Topology, method: PauliSynthStrat):
    initial_circ = pytket.Circuit(topo.num_qubits)
    circuit = gen_term_sequence_circuit(operator, initial_circ)

    Transform.UCCSynthesis(method, CXConfigType.Tree).apply(circuit)
    tket_arch = Architecture([(e1, e2) for (e1, e2) in topo.to_nx.edges()])
    unit = CompilationUnit(circuit)
    passes = SequencePass([
        PlacementPass(GraphPlacement(tket_arch)),
        RoutingPass(tket_arch),
    ])
    passes.apply(unit)
    circ_out = circuit
    Transform.DecomposeBoxes().apply(circ_out)
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


def synth_pp_tket_uccs_set(pp: PauliPolynomial, topo: Topology, prefix="tket_uccs_set"):
    circ_out = synth_tket(pp, topo, PauliSynthStrat.Sets)
    return get_ops_count(circ_out, prefix=prefix)


def synth_pp_tket_uccs_pair(pp: PauliPolynomial, topo: Topology, prefix="tket_uccs_pair"):
    circ_out = synth_tket(pp, topo, PauliSynthStrat.Pairwise)
    return get_ops_count(circ_out, prefix=prefix)


def synth_pp_pauliopt_ucc(pp: PauliPolynomial, topo: Topology, prefix="pauliopt_ucc"):
    synthesizer = PauliSynthesizer(pp, SynthMethod.UCCDS, topo)
    synthesizer.synthesize()
    return get_ops_count(synthesizer.circ_out_qiskit, prefix=prefix)


def synth_pp_pauliopt_steiner_nc(pp: PauliPolynomial, topo: Topology,
                                 prefix="pauliopt_steiner_nc"):
    pp = simplify_pauli_polynomial(pp, allow_acs=True)
    synthesizer = PauliSynthesizer(pp, SynthMethod.STEINER_GRAY_NC, topo)
    synthesizer.synthesize()
    return get_ops_count(synthesizer.circ_out_qiskit, prefix=prefix)


def synth_pp_pauliopt_divide_concquer(pp: PauliPolynomial, topo: Topology,
                                      prefix="pauliopt_divide_conquer"):
    synthesizer = PauliSynthesizer(pp, SynthMethod.DIVIDE_AND_CONQUER, topo)
    synthesizer.synthesize()
    return get_ops_count(synthesizer.circ_out_qiskit, prefix=prefix)


def synth_pp_naive(pp: PauliPolynomial, topo: Topology, prefix="naive"):
    circ_out = pp.to_circuit(topo).to_qiskit()
    return get_ops_count(circ_out, prefix=prefix)


def get_ops_count(qc: QuantumCircuit, prefix):
    if prefix == "":
        prefix = "circ"
    count = {f"{prefix}_cx": 0, f"{prefix}_depth": 0}
    ops = qc.count_ops()
    if "cx" in ops.keys():
        count[f"{prefix}_cx"] += ops['cx']
    if "swap" in ops.keys():
        count[f"{prefix}_cx"] += ops['swap'] * 3
    count[f"{prefix}_depth"] = qc.depth()
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


def get_topo_kind(topo_kind, num_qubits):
    if topo_kind == "line":
        return Topology.line(num_qubits)
    elif topo_kind == "complete":
        return Topology.complete(num_qubits)
    elif topo_kind == "cycle":
        return Topology.cycle(num_qubits)
    elif topo_kind == "grid":
        n_rows, n_cols = find_square_dimensions(num_qubits)
        return Topology.grid(n_rows, n_cols)
    else:
        raise Exception("Unknown topology kind")


def synth_ucc_evaluation():
    print("Start!")
    with open(f"{BASE_PATH}/orbital_lut.txt") as json_file:
        orbitals_lookup_table = json.load(json_file)
    for topo_kind in ["line", "complete", "cycle"]:
        for encoding_name in ["P", "BK", "JW"]:

            op_directory = f"{BASE_PATH}/operators/{encoding_name}_operators"
            results_file = f"data/pauli/uccsd/{topo_kind}/{encoding_name}_results.csv"
            print(encoding_name, flush=True)
            df = pd.DataFrame()
            for filename in os.listdir(op_directory):
                name = filename.replace(".pickle", "")
                print(name)
                active_spin_orbitals = orbitals_lookup_table[name]
                if encoding_name == "P":
                    n_qubits = active_spin_orbitals - 2
                else:
                    n_qubits = active_spin_orbitals
                print(n_qubits)
                if n_qubits >= 10:
                    continue
                path = op_directory + "/" + filename
                with open(path, "rb") as pickle_in:
                    qubit_pauli_operator = pickle.load(pickle_in)

                active_spin_orbitals = orbitals_lookup_table[name]
                if encoding_name == "P":
                    n_qubits = active_spin_orbitals - 2
                else:
                    n_qubits = active_spin_orbitals

                # if n_qubits >= 15:
                #     continue

                topo = get_topo_kind(topo_kind, n_qubits)
                pp = operator_to_pp(qubit_pauli_operator, n_qubits)
                col = {"name": name, "n_qubits": n_qubits, "gadgets": pp.num_gadgets}
                col = col | synth_pp_tket_uccs_set(pp, topo)
                col = col | synth_pp_tket_uccs_pair(pp, topo)
                col = col | synth_pp_pauliopt_steiner_nc(pp, topo)
                # col = col | synth_pp_pauliopt_divide_concquer(pp, topo)
                col = col | synth_pp_pauliopt_ucc(pp, topo)
                col = col | synth_pp_naive(pp, topo)
                df_col = pd.DataFrame(col, index=[0])
                print(df_col)
                df = pd.concat([df, df_col], ignore_index=True)
                df.to_csv(results_file)
            print("====")


def random_pauli_polynomial_experiment():
    df = pd.DataFrame()
    for num_gadgets in [10, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300, 500, 1000]:
        for num_qubits in [8]:
            for topo_name in ["complete"]:
                topo = get_topo_kind(topo_name, num_qubits)
                for _ in range(20):
                    pp = create_random_pauli_polynomial(num_qubits, num_gadgets)
                    naive_data = synth_pp_naive(pp, topo, prefix="naive")
                    pp = simplify_pauli_polynomial(pp, allow_acs=True)
                    for synth in ["tket_uccs_set", "tket_uccs_pair",
                                  "pauliopt_steiner_nc", "pauliopt_ucc"]:
                        col = {"method": synth, "n_qubits": num_qubits,
                               "gadgets": num_gadgets, "topo": topo_name} | naive_data

                        if synth == "tket_uccs_set":
                            col = col | synth_pp_tket_uccs_set(pp, topo, prefix="")
                        elif synth == "tket_uccs_pair":
                            col = col | synth_pp_tket_uccs_pair(pp, topo, prefix="")
                        elif synth == "pauliopt_steiner_nc":
                            col = col | synth_pp_pauliopt_steiner_nc(pp, topo, prefix="")
                        elif synth == "pauliopt_divide_conquer":
                            col = col | synth_pp_pauliopt_divide_concquer(pp, topo,
                                                                          prefix="")
                        elif synth == "pauliopt_ucc":
                            col = col | synth_pp_pauliopt_ucc(pp, topo, prefix="")
                        else:
                            raise Exception("Unknown synthesis method")

                        df_col = pd.DataFrame(col, index=[0])
                        print(df_col)
                        df = pd.concat([df, df_col], ignore_index=True)
                        df.to_csv("data/pauli/random/random_pauli_polynomial.csv")

    df.to_csv("data/pauli/random/random_pauli_polynomial.csv")


def fidelity_experiment_trotterisation():
    fid_default = []
    fid_ours = []
    fid_tket = []
    with open(f"{BASE_PATH}/orbital_lut.txt") as json_file:
        orbitals_lookup_table = json.load(json_file)

    logger = get_logger("fidelity_experiment_trotterisation")
    for name, encoding in [("H2_P_631g", "P"),
                           ("H4_P_sto3g", "P"),
                           ("LiH_P_sto3g", "P"),
                           ("LiH_JW_sto3g", "JW")]:

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
            summed_pauli_operator = operator_to_summed_pauli_op(operator, n_qubits)

            pp = summed_pauli_to_pp(summed_pauli_operator, n_qubits, t)

            evolution_op = (t / 2.0 * summed_pauli_operator).exp_i().to_matrix()

            U_expected = evolution_op  # qiskit.quantum_info.Operator(evolution_op)
            synthesizer = PauliSynthesizer(pp, SynthMethod.STEINER_GRAY_NC, topo)
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


def plot_random_pauli_polynomial_experiment():
    df = pd.read_csv("data/pauli/random/random_pauli_polynomial.csv")
    df = df[df["topo"] == "complete"]

    df["cx"] = (df["naive_cx"] - df["circ_cx"]) / df["naive_cx"]
    # plot by number of gadgets per method
    sns.barplot(x="gadgets", y="cx", hue="method", data=df)
    plt.show()


if __name__ == '__main__':
    fidelity_experiment_trotterisation()
    # synth_ucc_evaluation()
    # random_pauli_polynomial_experiment()
    # plot_random_pauli_polynomial_experiment()
