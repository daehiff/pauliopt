import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import pytket
import qiskit.quantum_info
import seaborn as sns
from matplotlib import pyplot as plt
from pytket._tket.transform import Transform
from pytket.extensions.qiskit import tk_to_qiskit
from pytket.utils import gen_term_sequence_circuit

from pauli_experiment import (
    pp_to_operator,
    topology_to_ph_graph,
    paulihedral_rep_from_paulipolynomial,
)
from paulihedral import synthesis_SC
from paulihedral.parallel_bl import depth_oriented_scheduling
from pauliopt.pauli.pauli_gadget import PauliGadget, PPhase
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.synthesis import PauliSynthesizer, SynthMethod
from pauliopt.pauli.utils import Pauli as Pauliopt_Pauli, apply_permutation
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
        elif algorithm == "paulihedral":
            graph = topology_to_ph_graph(topo)
            parr = paulihedral_rep_from_paulipolynomial(pp_.copy())

            lnq = len(parr[0][0])
            length = lnq // 2

            a1 = depth_oriented_scheduling(parr, length=length, maxiter=100)

            circ_out = synthesis_SC.block_opt_SC(a1, graph=graph, arch=None)

        else:
            raise Exception(f"Unknown Method: {algorithm}")

        H = pp_to_summed_pauli_op(pp_)
        U_expected = (t / 2.0 * H).exp_i().to_matrix()
        U_circ = qiskit.quantum_info.Operator.from_circuit(circ_out).data

        overlap = np.abs(ket_zero.conj().T @ U_circ.conj().T @ U_expected @ ket_zero)
        fid.append(overlap)
    return fid


def get_save_path(n_qubits, n_gadgets, algorithm):
    return f"data/fidelity_experiments/results/{n_qubits}_{n_gadgets}_{algorithm}"


def get_pp_save_path(i, n_qubits, n_gadgets):
    return f"data/fidelity_experiments/pp/{n_qubits}_{n_gadgets}_{i}"


def run_pp_experiment(n_qubits, n_gadgets, algorithm, i):
    start = time.time()
    t_start = 0.0
    t_end = 2 * np.pi
    t_steps = 50
    save_path = f"{get_pp_save_path(i, n_qubits, n_gadgets)}.pickle"
    assert os.path.isfile(save_path)
    with open(save_path, "rb") as f:
        pp = pickle.load(f)

    fids = get_fidelities(pp, algorithm, t_start, t_end, t_steps)
    print(fids)
    np.save(
        f"{get_save_path(n_qubits, n_gadgets, algorithm)}_{i}.npy",
        np.asarray(fids),
    )
    print(i, "Done", time.time() - start)
    return fids


def run_fidelity_experiment(algorithm, n_qubits, n_gadgets):
    args = [(n_qubits, n_gadgets, algorithm, i) for i in range(20)]

    with Pool(os.cpu_count()) as p:
        all_fidelities = p.starmap(run_pp_experiment, args)

    # all_fidelities = []
    # for arg in args:
    #     all_fidelities.append(run_pp_experiment(arg[0], arg[1], arg[2], arg[3]))

    np.save(
        f"{get_save_path(n_qubits, n_gadgets, algorithm)}.npy",
        np.asarray(all_fidelities),
    )


def run_fidelity_experiment_molecule(
    algorithm, molecule_name, t_start=0.0, t_end=2 * np.pi, t_steps=100
):
    pp_source_file = f"./datasets/pp_molecules/{molecule_name}.pickle"
    with open(pp_source_file, "rb") as pickle_in:
        pp = pickle.load(pickle_in)

    all_fidelities = get_fidelities(pp, algorithm, t_start, t_end, t_steps)
    np.save(
        f"data/fidelity_experiments/results/{molecule_name}_{algorithm}.npy",
        np.asarray(all_fidelities),
    )


def plot_fidelites(
    algorithms,
    molecules,
    n_qubits,
    n_gadgets,
    t_start=0.0,
    t_end=np.pi / 2.0,
    t_steps=50,
    p=90,
):
    plt.rcParams.update(
        {"text.usetex": False, "font.family": "sans-serif", "font.size": 16}
    )

    linestyles = [":", "-."]
    colors = palette = ['#377eb8', '#ff7f00', '#4daf4a',
               '#f781bf', '#a65628', '#984ea3',
               '#999999', '#e41a1c', '#dede00']
    ALG_NAMES = {
        "paulihedral": "Paulihedral",
        "PSGS": "Proposed",
        "default": "Naive Steiner tree decomp.",
        "UCCSD": "TKET UCCSD (set)",
    }

    fig, axes = plt.subplots(figsize=(13, 7), nrows=3, sharex=True)

    for color, algorithm in zip(colors, algorithms):
        all_fidelities = np.load(f"{get_save_path(n_qubits, n_gadgets, algorithm)}.npy")

        all_fidelities = np.asarray(all_fidelities)
        mean_fid = np.mean(all_fidelities, axis=0)
        ci_lower = np.percentile(all_fidelities, 100 - p, axis=0)
        ci_upper = np.percentile(all_fidelities, p, axis=0)
        time = np.linspace(t_start, t_end, t_steps)

        axes[0].plot(time, mean_fid, color=color)
        axes[0].fill_between(time, ci_upper, ci_lower, color=color, alpha=0.1)
    axes[0].set_title("Random Pauli Polynomials", fontsize=18)
    axes[0].set_ylabel("Unitary Overlap", fontsize=16)
    idx = 1
    for molecule_name, linestyle in zip(molecules, linestyles):
        for color, algorithm in zip(colors, algorithms):
            all_fidelities = np.load(
                f"data/fidelity_experiments/results/{molecule_name}_{algorithm}.npy"
            )

            all_fidelities = np.asarray(all_fidelities)

            time = np.linspace(t_start, t_end, t_steps)

            axes[idx].plot(
                time,
                all_fidelities[0],
                color=color,
                # label=f"{molecule_name}: {ALG_NAMES[algorithm]}",
            )
        axes[idx].set_title(f"{molecule_name}")
        axes[idx].set_ylabel("Unitary Overlap")
        idx += 1

    plt.xlabel("t", fontsize=16)
    plt.xticks(
        np.linspace(0.0, np.pi/2.0, 3),
        [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$"],
    )

    linestyles = ["solid"] + linestyles
    dummy_lines = []
    for b_idx in range(len(linestyles)):
        dummy_lines.append(axes[0].plot([], [], c="black", ls=linestyles[b_idx])[0])

    lines = axes[0].get_lines()
    fig.legend(
        [lines[i] for i in range(len(algorithms))],
        [ALG_NAMES[alg] for alg in algorithms],
        ncol=4,
        loc="upper center",
        borderaxespad=0
    )
    # axes[0].add_artist(legend1)
    #plt.tight_layout()
    plt.savefig(f"data/fidelity_experiments/plots/{n_qubits}_{n_gadgets}.pdf")
    #plt.show()


def molecule_fidelity_experiment(t_start=0.0, t_end=2 * np.pi, t_steps=50):
    algorithms = ["paulihedral", "PSGS", "default", "UCCSD"]
    for molecule_name in ["H2_P_631g", "H4_P_sto3g"]:
        save_path = f"datasets/pp_molecules/{molecule_name}.pickle"
        with open(save_path, "rb") as f:
            pp = pickle.load(f)

        allowed_angles = [np.pi / 32, np.pi / 64, np.pi / 128]
        pp.set_random_angles(allowed_angles)

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


def run_pp_experiments():
    algorithm = os.getenv("ALG")
    qubits = int(os.getenv("QUBITS"))
    gadgets = int(os.getenv("GADGETS"))
    print("Running: ", algorithm, qubits, gadgets)
    run_fidelity_experiment(algorithm, qubits, gadgets)


def generate_pps(n_qubits, n_gadgets):
    allowed_angels = [pi / 32, pi / 64, pi / 128]
    for i in range(20):
        save_path = f"{get_pp_save_path(i, n_qubits, n_gadgets)}.pickle"
        if not os.path.isfile(save_path):
            pp = generate_random_pauli_polynomial(
                n_qubits, n_gadgets, allowed_angels=allowed_angels
            )
            with open(save_path, "wb") as f:
                pickle.dump(pp, f)


if __name__ == "__main__":
    # generate_pps(4, 100)
    # generate_pps(6, 160)
    # generate_pps(10, 630)

    #run_pp_experiments()

    # molecule_fidelity_experiment()

    algorithms = ["paulihedral", "PSGS", "default", "UCCSD"]
    plot_fidelites(algorithms, ["H2_P_631g", "H4_P_sto3g"], 6, 160)

    # plot_fidelites(["paulihedral", "PSGS", "default", "UCCSD"], 6, 160)
    # plot_fidelites(["PSGS", "default", "UCCSD"], 10, 630)
