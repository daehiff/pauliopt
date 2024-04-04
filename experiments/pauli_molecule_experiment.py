import json
import logging
import os
import pickle
import shutil
from datetime import datetime
from numbers import Number

import matplotlib.pyplot as plt
import pandas as pd
import qiskit.quantum_info
import seaborn as sns
import sympy
from pytket.extensions.qiskit import qiskit_to_tk
from pytket.extensions.qiskit.backends import aer
from pytket.utils import QubitPauliOperator
from qiskit import QuantumCircuit
from qiskit.quantum_info import process_fidelity
from sympy import Symbol

from experiments.pauli_experiment import (
    term_sequence_tket,
    SYNTHESIS_METHODS,
    get_backend_and_df_name,
)
from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import *
from pauliopt.pauli.synthesis import PauliSynthesizer, SynthMethod
from pauliopt.utils import pi, AngleVar, Angle


def get_2q_depth(qc: QuantumCircuit):
    q = qiskit_to_tk(qc)
    return q.depth_2q()


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f"{name}.log")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def create_random_phase_gadget(
    num_qubits, min_legs, max_legs, allowed_angels, allowed_legs=None
):
    if allowed_legs is None:
        allowed_legs = [X, Y, Z]
    angle = np.random.choice(allowed_angels)
    nr_legs = np.random.randint(min_legs, max_legs)
    legs = np.random.choice([i for i in range(num_qubits)], size=nr_legs, replace=False)
    phase_gadget = [I for _ in range(num_qubits)]
    for leg in legs:
        phase_gadget[leg] = np.random.choice(allowed_legs)
    return PPhase(angle) @ phase_gadget


def create_random_pauli_polynomial(
    num_qubits: int, num_gadgets: int, min_legs=None, max_legs=None, allowed_angels=None
):
    if min_legs is None:
        min_legs = 1
    if max_legs is None:
        max_legs = num_qubits
    if allowed_angels is None:
        allowed_angels = [pi, pi / 2, pi / 4, pi / 8, pi / 16]

    pp = PauliPolynomial(num_qubits)
    for _ in range(num_gadgets):
        pp >>= create_random_phase_gadget(
            num_qubits, min_legs, max_legs, allowed_angels
        )

    return pp


def pp_to_operator(pp: PauliPolynomial):
    operator_list = []
    for l in range(pp.num_gadgets):
        temp_op = {}
        p_string = [
            [["q", [q]], pp[l][q].value] for q in range(pp.num_qubits) if pp[l][q] != I
        ]
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
        temp_op["string"] = [
            [["q", [q]], str(p)] for q, p in enumerate(pauli) if str(p) != "I"
        ]
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
        if num_qubits == 6:
            return Topology.grid(2, 3)
        elif num_qubits == 8:
            return Topology.grid(2, 4)
        else:
            n_rows, n_cols = find_square_dimensions(num_qubits)
            return Topology.grid(n_rows, n_cols)
    else:
        raise Exception("Unknown topology kind")


def create_csv_header():
    header = [
        "name",
        "num_qubits",
        "n_gadgets",
        "method",
        "cx",
        "u3",
        "depth",
        "2q_depth",
        "time",
    ]
    return header


def create_csv_header_real_hw():
    header = [
        "name",
        "backend",
        "num_qubits",
        "n_gadgets",
        "method",
        "cx",
        "u3",
        "depth",
        "2q_depth",
        "time",
    ]
    return header


def get_suitable_ibm_backend(n_qubits):
    available_backends = [
        ("quito", 5),
        ("nairobi", 7),
        ("guadalupe", 15),
        ("mumbai", 27),
        ("ithaca", 65),
        ("brisbane", 127),
    ]

    available_backends = list(sorted(available_backends, key=lambda x: x[1]))

    for name, backend_qubits in available_backends:
        if backend_qubits >= n_qubits:
            return name

    raise Exception(f"No backend with: {n_qubits} in list: {available_backends}")


def pad_pp_to_ibm_backend(pp: PauliPolynomial, n_qubits):
    pp_ = PauliPolynomial(n_qubits)

    pp_qubits = pp.num_qubits

    identity_pad = [I for _ in range(n_qubits - pp_qubits)]

    for gadget in pp:
        assert isinstance(gadget, PauliGadget)
        angle = gadget.angle
        paulis = gadget.paulis + identity_pad

        pp_ >>= PauliGadget(angle, paulis)
    return pp_


def real_hw_ucc_evaluation(max_qubits=7):
    logger = get_logger("real_hw_ucc_evaluation")
    op_directory = f"./datasets/pp_molecules/"
    for filename in os.listdir(op_directory):
        results_file = f"data/pauli/uccsd/ibm/results.csv"
        df = pd.DataFrame({c: [] for c in create_csv_header_real_hw()})

        name = filename.replace(".pickle", "")
        logger.info(name)
        path = op_directory + "/" + filename
        with open(path, "rb") as pickle_in:
            pp = pickle.load(pickle_in)
            pp = simplify_pauli_polynomial(pp, allow_acs=True)

        n_qubits = pp.num_qubits
        if n_qubits >= max_qubits:
            continue

        backend_name = get_suitable_ibm_backend(n_qubits)
        logger.info(f"Used {backend_name} with {n_qubits} on the PP")
        backend, _ = get_backend_and_df_name(backend_name)

        topo = Topology.from_qiskit_backend(backend)
        pp_ = pad_pp_to_ibm_backend(pp, topo.num_qubits)

        for synth_name, synth_method in SYNTHESIS_METHODS.items():
            start = datetime.now()
            count_dict = synth_method(pp_, topo)
            time_passed = (datetime.now() - start).total_seconds()
            column = {
                "name": name,
                "backend": backend_name,
                "num_qubits": n_qubits,
                "n_gadgets": pp.num_gadgets,
                "method": synth_name,
                "time": time_passed,
            } | count_dict
            print(column)
            new_row = pd.DataFrame([{c: column[c] for c in create_csv_header_real_hw()}])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(results_file)

def synth_ucc_evaluation(max_qubits=7):
    logger = get_logger("synth_ucc_evaluation")
    for topo_kind in ["line", "complete", "cycle"]:
        for encoding_name in ["P", "BK", "JW"]:
            op_directory = f"./datasets/pp_molecules/"
            results_file = f"data/pauli/uccsd/{topo_kind}/results.csv"
            df = pd.DataFrame({c: [] for c in create_csv_header()})
            with open(results_file, "wb") as f:
                df.to_csv(f, header=create_csv_header(), index=False)
            for filename in os.listdir(op_directory):
                name = filename.replace(".pickle", "")
                logger.info(name)
                path = op_directory + "/" + filename
                with open(path, "rb") as pickle_in:
                    pp = pickle.load(pickle_in)
                    pp = simplify_pauli_polynomial(pp, allow_acs=True)

                n_qubits = pp.num_qubits
                if n_qubits >= max_qubits:
                    continue

                topo = get_topo_kind(topo_kind, n_qubits)
                for synth_name, synth_method in SYNTHESIS_METHODS.items():
                    start = datetime.now()
                    count_dict = synth_method(pp, topo)
                    time_passed = (datetime.now() - start).total_seconds()
                    column = {
                        "name": name,
                        "num_qubits": n_qubits,
                        "n_gadgets": pp.num_gadgets,
                        "method": synth_name,
                        "time": time_passed,
                    } | count_dict

                    df = pd.DataFrame([{c: column[c] for c in create_csv_header()}])
                    with open(results_file, "ab") as f_ptr:
                        df.to_csv(f_ptr, header=False, index=False)
            print("====")


def fidelity_experiment_trotterisation():
    with open(f"{BASE_PATH}/orbital_lut.txt") as json_file:
        orbitals_lookup_table = json.load(json_file)

    logger = get_logger("fidelity_experiment_trotterisation")
    for name, encoding in [
        ("H2_P_631g", "P"),
        ("H4_P_sto3g", "P"),
        ("LiH_P_sto3g", "P"),
        ("LiH_JW_sto3g", "JW"),
    ]:
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
                "rb",
            ) as pickle_in:
                operator = pickle.load(pickle_in)

            topo = Topology.complete(n_qubits)
            summed_pauli_operator = operator_to_summed_pauli_op(operator, n_qubits)

            pp = summed_pauli_to_pp(summed_pauli_operator, n_qubits, t)

            evolution_op = (t / 2.0 * summed_pauli_operator).exp_i().to_matrix()

            # qiskit.quantum_info.Operator(evolution_op)
            U_expected = evolution_op
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
        df = pd.DataFrame(
            {
                "t": np.linspace(0.0, 1.0, 40),
                "fid_default": fid_default,
                "fid_ours": fid_ours,
                "fid_tket": fid_tket,
            }
        )
        df.to_csv(f"data/pauli/fidelity/{name}.csv")


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


def read_out_pps_tket_benchmarking():
    """
    Small convenience function to convert the experiments of Cowtan et. al to our datatype
    """
    with open(f"{BASE_PATH}/orbital_lut.txt") as json_file:
        orbitals_lookup_table = json.load(json_file)

    for encoding_name in ["P", "BK", "JW"]:
        base_dir = (
            f"tket_benchmarking/compilation_strategy/"
            f"operators/{encoding_name}_operators"
        )
        for filename in os.listdir(base_dir):
            name = filename.replace(".pickle", "")
            file_path = f"{base_dir}/{filename}"
            print(file_path)
            with open(file_path, "rb") as pickle_in:
                qubit_pauli_operator = pickle.load(pickle_in)

            active_spin_orbitals = orbitals_lookup_table[name]
            if encoding_name == "P":
                n_qubits = active_spin_orbitals - 2
            else:
                n_qubits = active_spin_orbitals

            pp = operator_to_pp(qubit_pauli_operator, n_qubits)

            op_directory = f"./datasets/pp_molecules/"

            with open(f"{op_directory}/{filename}", "wb") as f:
                pickle.dump(pp, f)


if __name__ == "__main__":
    real_hw_ucc_evaluation()
    # synth_ucc_evaluation()
    # read_out_pps_tket_benchmarking()
