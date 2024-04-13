import os
import pickle

import pytket
import qiskit.quantum_info
import scipy
from matplotlib import pyplot as plt
from pytket._tket.transform import PauliSynthStrat, Transform
from pytket.extensions.qiskit import tk_to_qiskit
from pytket.utils import gen_term_sequence_circuit
from qiskit.quantum_info import process_fidelity, state_fidelity, DensityMatrix

from pauli_experiment import pp_to_operator, synth_tket
from pauliopt.pauli.pauli_gadget import PauliGadget, PPhase
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.utils import Pauli as Pauliopt_Pauli, apply_permutation
import numpy as np

from pauliopt.pauli.synthesis import PauliSynthesizer, SynthMethod
from pauliopt.topologies import Topology
from pauliopt.utils import pi


def create_random_phase_gadget(
    num_qubits, min_legs, max_legs, allowed_angels, allowed_legs=None
):
    if allowed_legs is None:
        allowed_legs = [Pauliopt_Pauli.X, Pauliopt_Pauli.Y, Pauliopt_Pauli.Z]
    angle = np.random.choice(allowed_angels)
    nr_legs = np.random.randint(min_legs, max_legs)
    legs = np.random.choice([i for i in range(num_qubits)], size=nr_legs, replace=False)
    phase_gadget = [Pauliopt_Pauli.I for _ in range(num_qubits)]
    for leg in legs:
        phase_gadget[leg] = np.random.choice(allowed_legs)
    return PPhase(angle) @ phase_gadget


def generate_random_pauli_polynomial(
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


def pp_to_summed_pauli_op(pp: PauliPolynomial):
    from qiskit.opflow import I, X, Y, Z

    sum_pauli_op = 0.0
    for gadget in pp:
        assert isinstance(gadget, PauliGadget)
        coeff = gadget.angle

        paulis = [I for _ in range(pp.num_qubits)]
        for qb, pauli in enumerate(gadget.paulis):
            if pauli == Pauliopt_Pauli.I:
                continue
            elif pauli == Pauliopt_Pauli.X:
                paulis[qb] = X
            elif pauli == Pauliopt_Pauli.Y:
                paulis[qb] = Y
            elif pauli == Pauliopt_Pauli.Z:
                paulis[qb] = Z
        pauli_op = paulis[0]
        for pauli in paulis[1:]:
            pauli_op = pauli_op ^ pauli

        sum_pauli_op += float(coeff) * pauli_op
    return sum_pauli_op


def get_fidelities(pp: PauliPolynomial, algorithm, t_start, t_end, t_steps):
    fid = []

    ket_zero = np.zeros((2**pp.num_qubits,))
    ket_zero[0] = 1.0

    for t in np.linspace(t_start, t_end, t_steps):
        pp_ = pp.copy()

        t = float(t)
        pp_.assign_time(t)

        topo = Topology.complete(pp.num_qubits)
        if algorithm == "PSGS":
            synthesizer = PauliSynthesizer(pp_, SynthMethod.STEINER_GRAY_CLIFFORD, topo)
            synthesizer.synthesize()
            circ_out = synthesizer.circ_out_qiskit
            circ_out = apply_permutation(circ_out, synthesizer.qubit_placement)
        elif algorithm == "UCCSD":
            operator = pp_to_operator(pp_)

            initial_circ = pytket.Circuit(pp_.num_qubits)
            circ_out = gen_term_sequence_circuit(operator, initial_circ)

            Transform.DecomposeBoxes().apply(circ_out)
            Transform.RebaseToCliffordSingles().apply(circ_out)
            Transform.RebaseToRzRx().apply(circ_out)
            circ_out = tk_to_qiskit(circ_out)
        elif algorithm == "default":
            circ_out = pp_.to_qiskit(time=t)
        else:
            raise Exception(f"Unknown Method: {algorithm}")

        H = pp_to_summed_pauli_op(pp_)
        U_expected = (t / 2.0 * H).exp_i().to_matrix()
        U_circ = qiskit.quantum_info.Operator.from_circuit(circ_out).data

        overlap = np.abs(ket_zero.conj().T @ U_circ.conj().T @ U_expected @ ket_zero)
        # print(overlap)

        # fidelity = process_fidelity(circ_out, target=U_expected)
        # print(fidelity)
        # print("===")
        fid.append(overlap)
    return fid


def get_save_path(n_qubits, n_gadgets, algorithm):
    return f"data/fidelity_experiments/results/{n_qubits}_{n_gadgets}_{algorithm}"


def get_pp_save_path(i, n_qubits, n_gadgets):
    return f"data/fidelity_experiments/pp/{n_qubits}_{n_gadgets}_{i}"


def run_fidelity_experiment(
    algorithm, n_qubits, n_gadgets, t_start=0.0, t_end=2 * np.pi, t_steps=100
):
    allowed_angels = [pi / 16, pi / 32, pi / 64]

    all_fidelities = []

    for i in range(100):
        save_path = f"{get_pp_save_path(i, n_qubits, n_gadgets)}.pickle"
        print(os.path.isfile(save_path))
        if not os.path.isfile(save_path):
            pp = generate_random_pauli_polynomial(
                n_qubits, n_gadgets, allowed_angels=allowed_angels
            )
            with open(save_path, "wb") as f:
                pickle.dump(pp, f)
        else:
            with open(save_path, "rb") as f:
                pp = pickle.load(f)

        all_fidelities.append(get_fidelities(pp, algorithm, t_start, t_end, t_steps))

    all_fidelities = np.asarray(all_fidelities)

    np.save(f"{get_save_path(n_qubits, n_gadgets, algorithm)}.npy", all_fidelities)


def plot_fidelites(
    algorithms, n_qubits, n_gadgets, t_start=0.0, t_end=2 * np.pi, t_steps=100, p=90
):
    plt.rcParams.update(
        {"text.usetex": True, "font.family": "sans-serif", "font.size": 11}
    )

    colors = ["red", "blue", "green"]

    for color, algorithm in zip(colors, algorithms):
        all_fidelities = np.load(f"{get_save_path(n_qubits, n_gadgets, algorithm)}.npy")

        all_fidelities = np.asarray(all_fidelities)
        mean_fid = np.mean(all_fidelities, axis=0)
        ci_lower = np.percentile(all_fidelities, 100 - p, axis=0)
        ci_upper = np.percentile(all_fidelities, p, axis=0)
        time = np.linspace(t_start, t_end, t_steps)

        plt.plot(time, mean_fid, color=color, label=f"{algorithm}")
        plt.fill_between(time, ci_upper, ci_lower, color=color, alpha=0.1)

    plt.xlabel("t")
    plt.ylabel("Fidelity")
    plt.xticks(
        np.linspace(0.0, 2 * np.pi, 5),
        [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
    )
    plt.legend()
    plt.title("Median Fidelity")
    plt.show()


def molecule_fidelity_experiment(t_start=0.0, t_end=2 * np.pi, t_steps=100):
    algorithms = ["PSGS", "default", "UCCSD"]
    for molecule_name in ["H2_P_631g", "H4_P_sto3g", "LiH_P_sto3g"]:
        save_path = f"datasets/pp_molecules/{molecule_name}.pickle"
        with open(save_path, "rb") as f:
            pp = pickle.load(f)

        for algorithm in algorithms:
            all_fidelities = []
            all_fidelities.append(
                get_fidelities(pp, algorithm, t_start, t_end, t_steps)
            )

            save_path_data = (
                f"data/fidelity_experiments/results/{molecule_name}_{algorithm}"
            )

            np.save(save_path_data, all_fidelities)
        print(pp.num_qubits)
        print(pp.num_gadgets)


if __name__ == "__main__":
    # molecule_fidelity_experiment()

    run_fidelity_experiment("PSGS", 6, 160)
    run_fidelity_experiment("default", 6, 160)
    run_fidelity_experiment("UCCSD", 6, 160)

    # run_fidelity_experiment("PSGS", 10, 630)
    # run_fidelity_experiment("default", 10, 630)
    # run_fidelity_experiment("UCCSD", 10, 630)

    plot_fidelites(["PSGS"], 3, 160)
    # T = np.random.randn(100, 10) + np.linspace(0, 10, 100).reshape(-1, 1)
    # print(T.shape)
    # main()
