import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from qiskit import QuantumCircuit, transpile, execute, Aer
from qiskit.providers import JobStatus
from qiskit.providers.fake_provider import FakeVigo, FakeMumbai, FakeGuadalupe, FakeQuito, \
    FakeNairobi
from qiskit.providers.ibmq import IBMQ
from qiskit.quantum_info import Clifford, hellinger_fidelity

from pauliopt.pauli.clifford_tableau import CliffordTableau
from pauliopt.pauli.utils import apply_permutation
from pauliopt.topologies import Topology


def random_hscx_circuit(nr_gates=20, nr_qubits=4):
    gate_choice = ["H", "CX"]
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


def get_ops_count(qc: QuantumCircuit):
    count = {"h": 0, "cx": 0, "s": 0}
    ops = qc.count_ops()
    if "cx" in ops.keys():
        count["cx"] += ops['cx']
    if "h" in ops.keys():
        count["h"] += ops["h"]
    if "s" in ops.keys():
        count["s"] += ops["s"]
    return count


def qiskit_tableau_compilation(circ: QuantumCircuit, backend):
    tableau = Clifford.from_circuit(circ)
    circ_out = tableau.to_circuit()
    circ_out = transpile(circ_out, backend=backend, basis_gates=["cx", "h", "s"])
    print("Qiskit Tableau: ", circ_out.count_ops())
    column = get_ops_count(circ_out)
    return column


def qiskit_compilation(circ: QuantumCircuit, backend):
    circ_out = transpile(circ, backend=backend, basis_gates=["cx", "h", "s"])
    column = get_ops_count(circ_out)
    print("Qiskit: ", circ_out.count_ops())
    return column


def our_compilation(circ: QuantumCircuit, backend):
    topo = Topology.from_qiskit_backend(backend)
    ct = CliffordTableau.from_circuit(circ)
    circ_out, _ = ct.to_cifford_circuit_arch_aware(topo, include_swaps=True)
    circ_out = circ_out.to_qiskit()
    column = get_ops_count(circ_out)
    # assert verify_equality(circ, circ_out)
    print("Our: ", circ_out.count_ops())
    return column


def our_compilation_tableau(tab: Clifford, backend):
    topo = Topology.from_qiskit_backend(backend)
    ct = CliffordTableau.from_qiskit(tab)
    circ_out, perm = ct.to_cifford_circuit_arch_aware(topo)
    column = get_ops_count(circ_out)
    # assert verify_equality(tab.to_circuit(), apply_permutation(circ_out, perm))
    print("Our: ", circ_out.count_ops())
    return column


def random_experiment(backend_name="vigo", nr_input_gates=100, nr_steps=5):
    df_name = "data/random"
    if backend_name == "vigo":
        backend = FakeVigo()
        df_name = f"{df_name}_vigo.csv"
    elif backend_name == "mumbai":
        backend = FakeMumbai()
        df_name = f"{df_name}_mumbai.csv"
    elif backend_name == "guadalupe":
        backend = FakeGuadalupe()
        df_name = f"{df_name}_guadalupe.csv"
    elif backend_name == "quito":
        backend = FakeQuito()
        df_name = f"{df_name}_quito.csv"
    elif backend_name == "nairobi":
        backend = FakeNairobi()
        df_name = f"{df_name}_nairobi.csv"
    else:
        raise ValueError(f"Unknown backend: {backend_name}")
    num_qubits = backend.configuration().num_qubits
    print(df_name)
    print(num_qubits)
    df = pd.DataFrame(
        columns=["n_rep", "num_qubits", "n_gadgets", "method", "h", "s", "cx"])
    for n_gadgets in range(1, nr_input_gates, nr_steps):
        print(n_gadgets)
        for _ in range(20):
            circ = random_hscx_circuit(nr_qubits=num_qubits, nr_gates=n_gadgets)

            ########################################
            # Our clifford circuit
            ########################################
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "method": "tableau"} | \
                     our_compilation(circ, backend)
            df.loc[len(df)] = column
            df.to_csv(df_name)

            # ########################################
            # # Qiskit compilation
            # ########################################
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "method": "qiskit"} | \
                     qiskit_compilation(circ, backend)
            df.loc[len(df)] = column
            df.to_csv(df_name)

            ########################################
            # Qiskit tableau compilation
            ########################################
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "method": "qiskit_tableau"} | \
                     qiskit_tableau_compilation(circ, backend)
            df.loc[len(df)] = column
            df.to_csv(df_name)

    df.to_csv(df_name)
    # print(circ_out)
    # print(circ_stim)


def apply_permutation_measurements(counts: dict, perm: list):
    new_counts = {}
    for key, value in counts.items():
        new_key = "".join([key[i] for i in perm])
        new_counts[new_key] = value
    return new_counts


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    return str(obj)


def run_clifford_experiment(exp_number=0):
    if not os.path.exists(
            f"experiments/data/clifford_experiment_real_hardware/{exp_number}"):
        os.makedirs(f"experiments/data/clifford_experiment_real_hardware/{exp_number}")

    # read token from env IBM_TOKEN

    token = os.environ.get("IBM_TOKEN")
    assert token
    provider = IBMQ.enable_account(token)
    backend = provider.get_backend('ibmq_quito')

    # Get the provider and choose the backend
    clifford_circ = random_hscx_circuit(nr_gates=40,
                                        nr_qubits=backend.configuration().num_qubits)
    clifford = Clifford.from_circuit(clifford_circ)
    ct = CliffordTableau(tableau=clifford.symplectic_matrix, signs=clifford.phase)
    with open(
            f"experiments/data/clifford_experiment_real_hardware/{exp_number}/clifford.json",
            "w") as f:
        json.dump(clifford.to_dict(), f)

    backend_simulator = Aer.get_backend('statevector_simulator')
    circ_simulated = clifford.to_circuit()
    circ_simulated.measure_all()
    job = execute(circ_simulated, backend_simulator, shots=8000)
    result = job.result()
    counts_simulated = result.get_counts()
    print(counts_simulated)

    topo = Topology.from_qiskit_backend(backend)
    circ_ours, perm = ct.to_cifford_circuit_arch_aware(topo)

    circ_ours = apply_permutation(circ_ours, perm)
    circ_ours.name = f"clifford_synth_{exp_number}"

    circ_ours.measure_all()
    job_ours = execute(circ_ours, backend, shots=8000)

    circ = clifford.to_circuit()
    circ.measure_all()
    circ_ours.name = f"ibm_{exp_number}"
    circ = transpile(circ, backend)
    job_ibm = execute(circ, backend, shots=8000)

    print("Ours: ", circ_ours.count_ops())
    print("IBM: ", circ.count_ops())

    while job_ibm.status() != JobStatus.DONE:
        time.sleep(1)

    result_ibm = job_ibm.result()
    with open(
            f"experiments/data/clifford_experiment_real_hardware/{exp_number}/result_ibm.json",
            "w") as f:
        json.dump(result_ibm.to_dict(), f, default=json_serial)

    while job_ours.status() != JobStatus.DONE:
        time.sleep(1)
    result_ours = job_ours.result()
    with open(
            f"experiments/data/clifford_experiment_real_hardware/{exp_number}/result_ours.json",
            "w") as f:
        json.dump(result_ours.to_dict(), f, default=json_serial)

    count_ours = result_ours.get_counts()
    count_ours = apply_permutation_measurements(count_ours, perm)
    count_ibm = result_ibm.get_counts()
    fidelity_ours = hellinger_fidelity(count_ours, counts_simulated)
    fidelity_ibm = hellinger_fidelity(count_ibm, counts_simulated)

    print("Ours: ", fidelity_ours)
    print("IBM: ", fidelity_ibm)

    col = {
        "cx_ours": circ_ours.count_ops()["cx"],
        "cx_ibm": circ.count_ops()["cx"],
        "fidelity_ours": fidelity_ours,
        "fidelity_ibm": fidelity_ibm,
        "time_ours": result_ours.time_taken,
        "time_ibm": result_ibm.time_taken
    }
    return pd.DataFrame(col, index=[0])


def plot_vigo():
    df = pd.read_csv(f"data/random_vigo.csv")
    #
    sns.lineplot(df, x="n_gadgets", y="h", hue="method")
    plt.title("H-Gates (Vigo)")
    plt.xlabel("Number of input Gates")
    plt.ylabel("Number of H-Gates")
    plt.show()

    sns.lineplot(df, x="n_gadgets", y="s", hue="method")
    plt.title("S-Gates (Vigo)")
    plt.xlabel("Number of input gates")
    plt.ylabel("Number of S-Gates")
    plt.show()

    sns.lineplot(df, x="n_gadgets", y="cx", hue="method")
    plt.title("CNOT-Gates (Vigo9")
    plt.xlabel("Number of input Gates")
    plt.ylabel("Number of CNOT-Gates")
    plt.show()


def plot_guadalupe():
    df = pd.read_csv(f"data/random_guadalupe.csv")
    #
    sns.lineplot(df, x="n_gadgets", y="h", hue="method")
    plt.title("H-Gates (Guadalupe)")
    plt.xlabel("Number of input Gates")
    plt.ylabel("Number of H-Gates")
    plt.show()

    sns.lineplot(df, x="n_gadgets", y="s", hue="method")
    plt.title("S-Gates (Guadalupe)")
    plt.xlabel("Number of input gates")
    plt.ylabel("Number of S-Gates")
    plt.show()

    sns.lineplot(df, x="n_gadgets", y="cx", hue="method")
    plt.title("CNOT-Gates (Guadalupe)")
    plt.xlabel("Number of input Gates")
    plt.ylabel("Number of CNOT-Gates")
    plt.show()


def plot_mumbai():
    df = pd.read_csv(f"data/random_mumbai.csv")
    #
    sns.lineplot(df, x="n_gadgets", y="h", hue="method")
    plt.title("H-Gates (Mumbai)")
    plt.xlabel("Number of input Gates")
    plt.ylabel("Number of H-Gates")
    plt.show()

    sns.lineplot(df, x="n_gadgets", y="s", hue="method")
    plt.title("S-Gates (Mumbai)")
    plt.xlabel("Number of input gates")
    plt.ylabel("Number of S-Gates")
    plt.show()

    sns.lineplot(df, x="n_gadgets", y="cx", hue="method")
    plt.title("CNOT-Gates (Mumbai)")
    plt.xlabel("Number of input Gates")
    plt.ylabel("Number of CNOT-Gates")
    plt.show()


def plot_quito():
    df = pd.read_csv(f"data/random_quito.csv")

    sns.lineplot(df, x="n_gadgets", y="h", hue="method")
    plt.title("H-Gates (Quito)")
    plt.xlabel("Number of input Gates")
    plt.ylabel("Number of H-Gates")
    plt.show()

    sns.lineplot(df, x="n_gadgets", y="s", hue="method")
    plt.title("S-Gates (Quito)")
    plt.xlabel("Number of input gates")
    plt.ylabel("Number of S-Gates")
    plt.show()

    sns.lineplot(df, x="n_gadgets", y="cx", hue="method")
    plt.title("CNOT-Gates (Quito)")
    plt.xlabel("Number of input Gates")
    plt.ylabel("Number of CNOT-Gates")
    plt.show()


def plot_nairobi():
    df = pd.read_csv(f"data/random_nairobi.csv")

    sns.lineplot(df, x="n_gadgets", y="h", hue="method")
    plt.title("H-Gates (Nairobi)")
    plt.xlabel("Number of input Gates")
    plt.ylabel("Number of H-Gates")
    plt.show()

    sns.lineplot(df, x="n_gadgets", y="s", hue="method")
    plt.title("S-Gates (Nairobi)")
    plt.xlabel("Number of input gates")
    plt.ylabel("Number of S-Gates")
    plt.show()

    sns.lineplot(df, x="n_gadgets", y="cx", hue="method")
    plt.title("CNOT-Gates (Nairobi)")
    plt.xlabel("Number of input Gates")
    plt.ylabel("Number of CNOT-Gates")
    plt.show()


def run_clifford_real_hardware():
    experiment_numbers = [i for i in range(1, 20)]

    # create a threadpool and run run_clifford_experiment
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit tasks to the thread pool

        results = executor.map(run_clifford_experiment, experiment_numbers)
    df = pd.concat(results, ignore_index=True)
    df.to_csv("data/clifford_experiment_real_hardware/results.csv")


if __name__ == "__main__":
    run_clifford_real_hardware()
    # generate_cliffords(0)
    # random_experiment(backend_name="nairobi", nr_input_gates=70, nr_steps=5)

    # random_experiment(backend_name="quito", nr_input_gates=70, nr_steps=5)
    # random_experiment(backend_name="guadalupe", nr_input_gates=250, nr_steps=10)
    # random_experiment(backend_name="mumbai", nr_input_gates=600, nr_steps=20)
    # plot_mumbai()
    # plot_vigo()
    # plot_guadalupe()
    # plot_nairobi()
    # plot_quito()
