import functools
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import stim
from qiskit import QuantumCircuit, transpile, execute, Aer
from qiskit.providers import JobStatus
from qiskit.providers.fake_provider import FakeVigo, FakeMumbai, FakeGuadalupe, FakeQuito, \
    FakeNairobi, FakePerth, ConfigurableFakeBackend
from qiskit.providers.ibmq import IBMQ
from qiskit.quantum_info import Clifford, hellinger_fidelity
from qiskit.result import Result
from qiskit_ibm_provider import IBMProvider

from pauliopt.pauli.clifford_tableau import CliffordTableau
from pauliopt.pauli.utils import apply_permutation
from pauliopt.topologies import Topology


def random_hscx_circuit(nr_gates=20, nr_qubits=4):
    gate_choice = ["H", "S", "CX"]
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


def circuit_to_stim_tableau(circ: QuantumCircuit):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    tableau = stim.Tableau(circ.num_qubits)
    cnot = stim.Tableau.from_named_gate("CX")
    had = stim.Tableau.from_named_gate("H")
    s = stim.Tableau.from_named_gate("S")
    for op in circ:
        if op.operation.name == "h":
            tableau.append(had, [op.qubits[0].index])
        elif op.operation.name == "s":
            tableau.append(s, [op.qubits[0].index])
        elif op.operation.name == "cx":
            tableau.append(cnot, [op.qubits[0].index, op.qubits[1].index])
        else:
            raise Exception("Unknown operation")
    return tableau


def parse_stim_to_qiskit(circ: stim.Circuit):
    qc = QuantumCircuit(circ.num_qubits)
    for gate in circ:
        if gate.name == "CX":
            targets = [target.value for target in gate.targets_copy()]
            targets = [(targets[i], targets[i + 1]) for i in range(0, len(targets), 2)]
            for (ctrl, target) in targets:
                qc.cx(ctrl, target)
        elif gate.name == "H":
            targets = [target.value for target in gate.targets_copy()]
            for qubit in targets:
                qc.h(qubit)
        elif gate.name == "S":
            targets = [target.value for target in gate.targets_copy()]
            for qubit in targets:
                qc.s(qubit)
        else:
            raise TypeError(f"Unknown Name: {gate.name}")
    return qc


def stim_compilation(circ: QuantumCircuit, backend):
    start = time.time()
    tableau = circuit_to_stim_tableau(circ)
    circ_out = parse_stim_to_qiskit(tableau.to_circuit(method="elimination"))
    if backend == "complete":
        circ_out = transpile(circ_out, basis_gates=["cx", "h", "s"])
    else:
        circ_out = transpile(circ_out,
                             coupling_map=backend.coupling_map,
                             basis_gates=["cx", "h", "s"])
    print("Qiskit Tableau: ", circ_out.count_ops(), "Time: ", time.time() - start)
    column = get_ops_count(circ_out)
    return column


def qiskit_tableau_compilation(circ: QuantumCircuit, backend):
    start = time.time()
    tableau = Clifford.from_circuit(circ)
    circ_out = tableau.to_circuit()
    if backend == "complete":
        circ_out = transpile(circ_out, basis_gates=["cx", "h", "s"])
    else:
        circ_out = transpile(circ_out,
                             coupling_map=backend.coupling_map,
                             basis_gates=["cx", "h", "s"])
    print("Qiskit Tableau: ", circ_out.count_ops(), "Time: ", time.time() - start)
    column = get_ops_count(circ_out)
    return column


def qiskit_compilation(circ: QuantumCircuit, backend):
    start = time.time()
    if backend == "complete":
        circ_out = transpile(circ, basis_gates=['s', 'h', 'cx'])
    else:
        circ_out = transpile(circ, coupling_map=backend.coupling_map,
                             basis_gates=['s', 'h', 'cx'])
    column = get_ops_count(circ_out)
    print("Qiskit: ", circ_out.count_ops(), "Time: ", time.time() - start)
    return column


def our_compilation(circ: QuantumCircuit, backend):
    start = time.time()
    if backend == "complete":
        topo = Topology.complete(circ.num_qubits)
    else:
        topo = Topology.from_qiskit_backend(backend)
    ct = CliffordTableau.from_circuit(circ)
    circ_out, _ = ct.to_cifford_circuit_arch_aware(topo, include_swaps=True)
    circ_out = circ_out.to_qiskit()
    column = get_ops_count(circ_out)
    # assert verify_equality(circ, circ_out)
    print("Our: ", circ_out.count_ops(), "Time: ", time.time() - start)
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
    elif backend_name == "perth":
        backend = FakeNairobi()
        df_name = f"{df_name}_perth.csv"
    elif backend_name == "ithaca":
        backend = FakeJSONBackend("ibm_ithaca")
        df_name = f"{df_name}_ithaca.csv"
    elif backend_name == "seattle":
        backend = FakeJSONBackend("ibm_seattle")
        df_name = f"{df_name}_seattle.csv"
    elif backend_name == "brisbane":
        backend = FakeJSONBackend("ibm_brisbane")
        df_name = f"{df_name}_brisbane.csv"
    elif backend_name == "complete":
        backend = "complete"
        df_name = f"{df_name}_complete.csv"

    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    if backend != "complete":
        num_qubits = backend.configuration().num_qubits
    else:
        num_qubits = 16
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
                      "method": "ours"} | \
                     our_compilation(circ, backend)
            df.loc[len(df)] = column
            df.to_csv(df_name)

            # ########################################
            # # Qiskit compilation
            # ########################################
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "method": "qiskit transpile"} | \
                     qiskit_compilation(circ, backend)
            df.loc[len(df)] = column
            df.to_csv(df_name)

            ########################################
            # Qiskit tableau compilation
            ########################################
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "method": "Bravyi et al. (qiskit)"} | \
                     qiskit_tableau_compilation(circ, backend)
            df.loc[len(df)] = column
            df.to_csv(df_name)

            ########################################
            # Stim compilation
            ########################################
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "method": "Ewout van den Berg (stim)"} | \
                     stim_compilation(circ, backend)
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


def run_clifford_experiment(exp_number=0, backend_name="ibm_perth"):
    base_path = f"data/clifford_experiment_real_hardware/{backend_name}/{exp_number}"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # read token from env IBM_TOKEN
    token = os.environ.get("IBM_TOKEN")
    assert token
    # provider = IBMQ.enable_account(token)
    provider = IBMProvider(token=token, instance="ibm-q-research-2/tu-munich-2/main")
    backend = provider.get_backend(backend_name)
    if backend_name == "ibm_perth" or backend_name == "ibm_nairobi":
        nr_gates = 30
    else:
        nr_gates = 25
    # Get the provider and choose the backend
    clifford_circ = random_hscx_circuit(nr_gates=nr_gates,
                                        nr_qubits=backend.configuration().num_qubits)
    clifford = Clifford.from_circuit(clifford_circ)
    ct = CliffordTableau(tableau=clifford.symplectic_matrix, signs=clifford.phase)
    with open(f"{base_path}/clifford.json", "w") as f:
        json.dump(clifford.to_dict(), f)

    backend_simulator = Aer.get_backend('statevector_simulator')
    circ_simulated = clifford.to_circuit()
    circ_simulated_ = QuantumCircuit(circ_simulated.num_qubits,
                                     name=f"clifford_simulated_{exp_number}")
    circ_simulated_.compose(circ_simulated, inplace=True)
    circ_simulated_.barrier()
    circ_simulated_.compose(circ_simulated.inverse(), inplace=True)
    circ_simulated_.measure_all()
    circ_simulated = circ_simulated_
    circ_simulated.qasm(filename=f"{base_path}/clifford_simulated.qasm")
    job = execute(circ_simulated, backend_simulator, shots=16000)
    result = job.result()
    counts_simulated = result.get_counts()
    print(counts_simulated)

    ######################
    # Stim execution
    ######################

    tableau = circuit_to_stim_tableau(clifford_circ)
    circ_stim = parse_stim_to_qiskit(tableau.to_circuit(method="elimination"))
    circ_stim = transpile(circ_stim, backend)

    circ_stim_ = QuantumCircuit(circ_stim.num_qubits,
                                name=f"clifford_stim_{exp_number}")
    circ_stim_.compose(circ_stim, inplace=True)
    circ_stim_.barrier()
    circ_stim_.compose(circ_stim.inverse(), inplace=True)
    circ_stim_.measure_all()

    job_stim = execute(circ_stim_, backend, shots=16000)

    print("Stim: ", circ_stim_.count_ops()["cx"])

    while job_stim.status() != JobStatus.DONE:
        time.sleep(1)
    result_stim = job_stim.result()
    with open(f"{base_path}/result_stim.json", "w") as f:
        json.dump(result_stim.to_dict(), f, default=json_serial)

    circ_stim_.qasm(filename=f"{base_path}/stim.qasm")

    ######################
    # Our execution of the quantum circuit
    ######################

    topo = Topology.from_qiskit_backend(backend)
    circ_ours, perm = ct.to_cifford_circuit_arch_aware(topo)
    circ_ours = apply_permutation(circ_ours.to_qiskit(), perm)

    circ_ours_ = QuantumCircuit(circ_ours.num_qubits,
                                name=f"clifford_synth_{exp_number}")
    circ_ours_.compose(circ_ours, inplace=True)
    circ_ours_.barrier()
    circ_ours_.compose(circ_ours.inverse(), inplace=True)
    circ_ours_.measure_all()
    circ_ours = circ_ours_
    circ_ours.qasm(filename=f"{base_path}/ours.qasm")

    job_ours = execute(circ_ours, backend, shots=16000)

    ######################
    # IBM execution of the quantum circuit
    ######################
    circ = clifford.to_circuit()

    circ = transpile(circ, backend)

    circ_ = QuantumCircuit(circ_ours.num_qubits,
                           name=f"ibm_synth_{exp_number}")
    circ_.compose(circ, inplace=True)
    circ_.barrier()
    circ_.compose(circ.inverse(), inplace=True)
    circ_.measure_all()
    circ = circ_
    circ.qasm(filename=f"{base_path}/ibm.qasm")
    job_ibm = execute(circ, backend, shots=16000)

    print("Ours: ", circ_ours.count_ops()["cx"])
    print("IBM: ", circ.count_ops()["cx"])

    while job_ibm.status() != JobStatus.DONE:
        time.sleep(1)

    result_ibm = job_ibm.result()
    with open(f"{base_path}/result_ibm.json", "w") as f:
        json.dump(result_ibm.to_dict(), f, default=json_serial)

    while job_ours.status() != JobStatus.DONE:
        time.sleep(1)
    result_ours = job_ours.result()
    with open(f"{base_path}/result_ours.json", "w") as f:
        json.dump(result_ours.to_dict(), f, default=json_serial)

    count_ours = result_ours.get_counts()
    count_ours = apply_permutation_measurements(count_ours, perm)
    count_ibm = result_ibm.get_counts()
    count_stim = result_stim.get_counts()

    fidelity_ours = hellinger_fidelity(count_ours, counts_simulated)
    fidelity_ibm = hellinger_fidelity(count_ibm, counts_simulated)
    fidelity_stim = hellinger_fidelity(count_stim, counts_simulated)

    print("Ours: ", fidelity_ours)
    print("IBM: ", fidelity_ibm)
    print("Stim: ", fidelity_stim)

    col = {
        "cx_ours": circ_ours.count_ops()["cx"],
        "cx_ibm": circ.count_ops()["cx"],
        "fidelity_ours": fidelity_ours,
        "fidelity_ibm": fidelity_ibm,
        "time_ours": result_ours.time_taken,
        "time_ibm": result_ibm.time_taken
    }
    return pd.DataFrame(col, index=[0])


def plot_experiment(name="random_guadalupe"):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 11
    })
    sns.set_palette(sns.color_palette("colorblind"))

    df = pd.read_csv(f"data/{name}.csv")
    #
    sns.lineplot(df, x="n_gadgets", y="h", hue="method")
    plt.title("H-Gates")
    plt.xlabel("Number of input Gates")
    plt.ylabel("Number of H-Gates")
    plt.savefig(f"data/{name}_h.pdf")
    plt.show()

    sns.lineplot(df, x="n_gadgets", y="s", hue="method")
    plt.title("S-Gates")
    plt.xlabel("Number of input gates")
    plt.ylabel("Number of S-Gates")
    plt.savefig(f"data/{name}_s.pdf")
    plt.show()

    sns.lineplot(df, x="n_gadgets", y="cx", hue="method")
    plt.title("CNOT-Gates")
    plt.xlabel("Number of input Gates")
    plt.ylabel("Number of CNOT-Gates")
    plt.savefig(f"data/{name}_cx.pdf")
    plt.show()


def run_clifford_real_hardware(backend_name="ibm_perth"):
    experiment_numbers = [i for i in range(0, 20)]
    run_clifford_experiment_ = functools.partial(run_clifford_experiment,
                                                 backend_name=backend_name)
    # create a threadpool and run run_clifford_experiment
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit tasks to the thread pool

        results = executor.map(run_clifford_experiment_, experiment_numbers)
    df = pd.concat(results, ignore_index=True)
    print(df)
    df.to_csv("data/clifford_experiment_real_hardware/results.csv")


def analyze_real_hw(backend_name="ibm_nairobi"):
    cumult_counts_ibm = {}
    cumult_counts_ours = {}
    cumult_counts_stim = {}
    num_qubits = 1
    df = pd.DataFrame(
        columns=["method", "h", "s", "cx", "fid", "execution_time"])
    for i in range(0, 20):
        base_path = f"data/clifford_experiment_real_hardware/{backend_name}/{i}"

        with open(f"{base_path}/result_ibm.json", "r") as f:
            result = json.load(f)
            result_ibm = Result.from_dict(result)
        circ_ibm = QuantumCircuit.from_qasm_file(f"{base_path}/ibm.qasm")
        num_qubits = circ_ibm.num_qubits
        with open(f"{base_path}/result_ours.json", "r") as f:
            result = json.load(f)
            result_ours = Result.from_dict(result)
        circ_ours = QuantumCircuit.from_qasm_file(f"{base_path}/ours.qasm")

        with open(f"{base_path}/result_stim.json", "r") as f:
            result = json.load(f)
            result_stim = Result.from_dict(result)
        circ_stim = QuantumCircuit.from_qasm_file(f"{base_path}/stim.qasm")

        counts_expected = {"0" * num_qubits: 16000}
        counts_ibm = result_ibm.get_counts()
        counts_ours = result_ours.get_counts()
        counts_stim = result_stim.get_counts()
        fid_ibm = hellinger_fidelity(counts_expected, counts_ibm)
        fid_ours = hellinger_fidelity(counts_expected, counts_ours)
        fid_stim = hellinger_fidelity(counts_expected, counts_stim)

        column = {
            "method": "ibm",
            "h": circ_ibm.count_ops()["h"] if "h" in circ_ibm.count_ops() else 0,
            "s": circ_ibm.count_ops()["s"] if "s" in circ_ibm.count_ops() else 0,
            "cx": circ_ibm.count_ops()["cx"] if "cx" in circ_ibm.count_ops() else 0,
            "fid": fid_ibm,
            "execution_time": result_ibm.time_taken,
        }
        df.loc[len(df)] = column
        column = {
            "method": "ours",
            "h": circ_ours.count_ops()["h"] / 2.0
            if "h" in circ_ours.count_ops() else 0,
            "s": circ_ours.count_ops()["s"] / 2.0
            if "s" in circ_ours.count_ops() else 0,
            "cx": circ_ours.count_ops()["cx"] / 2.0
            if "cx" in circ_ours.count_ops() else 0,
            "fid": fid_ours,
            "execution_time": result_ours.time_taken,
        }
        df.loc[len(df)] = column

        column = {
            "method": "stim",
            "h": circ_stim.count_ops()["h"] / 2.0
            if "h" in circ_stim.count_ops() else 0,
            "s": circ_stim.count_ops()["s"] / 2.0
            if "s" in circ_stim.count_ops() else 0,
            "cx": circ_stim.count_ops()["cx"] / 2.0
            if "cx" in circ_stim.count_ops() else 0,
            "fid": fid_stim,
            "execution_time": result_stim.time_taken,
        }
        df.loc[len(df)] = column
        for key, value in counts_ibm.items():
            if key in cumult_counts_ibm:
                cumult_counts_ibm[key] += [value]
            else:
                cumult_counts_ibm[key] = [value]

        for key, value in counts_ours.items():
            if key in cumult_counts_ours:
                cumult_counts_ours[key] += [value]
            else:
                cumult_counts_ours[key] = [value]

        for key, value in counts_stim.items():
            if key in cumult_counts_stim:
                cumult_counts_stim[key] += [value]
            else:
                cumult_counts_stim[key] = [value]

    df.to_csv(f"data/clifford_experiment_real_hardware/{backend_name}/analysis.csv")

    for name in ["ours", "stim", "ibm"]:
        print(
            f"Mean execution time {name}: {df[df['method'] == name]['execution_time'].mean()}")

    df_ = df[df["method"] == "ours"]
    correlation_matrix = df_[["h", "s", "cx", "fid"]].corr()
    print("Correlation Matrix CNOTs:")
    print(correlation_matrix)
    print(df.groupby("method").mean())

    lables = []
    values = []
    for key, value in cumult_counts_ours.items():
        lables.append([key] * len(value))
        values.append(value)
    lables = list(np.hstack(lables))
    values = list(np.hstack(values))
    cat = ["ours"] * len(lables)
    df_our = pd.DataFrame({"approach": cat,
                           "lables": lables,
                           "values": values})

    lables = []
    values = []
    for key, value in cumult_counts_ibm.items():
        lables.append([key] * len(value))
        values.append(value)
    lables = list(np.hstack(lables))
    values = list(np.hstack(values))
    cat = ["Bravyi et al. (qiskit)"] * len(lables)
    df_ibm = pd.DataFrame({"approach": cat,
                           "lables": lables,
                           "values": values})

    lables = []
    values = []
    for key, value in cumult_counts_stim.items():
        lables.append([key] * len(value))
        values.append(value)

    lables = list(np.hstack(lables))
    values = list(np.hstack(values))
    cat = ["Ewout van den Berg (stim)"] * len(lables)
    df_stim = pd.DataFrame({"approach": cat,
                            "lables": lables,
                            "values": values})
    df = pd.concat([df_our, df_ibm, df_stim], ignore_index=True)

    # seaborn barplot rotate the x ticks labels by 90 degrees
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 11
    })
    sns.set_palette(sns.color_palette("colorblind"))
    sns.barplot(data=df, x="lables", y="values", hue="approach", errwidth=0.1)
    plt.title("Average counts for 20 random Clifford circuits")
    plt.xticks(rotation=90)
    plt.xlabel("Measured State")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig("data/clifford_experiment_real_hardware/counts_clifford.pdf")
    plt.show()

    counts_expected = {'0' * num_qubits: 16000}
    averga_counts_ours = {key: np.mean(value) for key, value in
                          cumult_counts_ours.items()}
    averga_counts_ibm = {key: np.mean(value) for key, value in cumult_counts_ibm.items()}
    averga_counts_stim = {key: np.mean(value) for key, value in
                          cumult_counts_stim.items()}
    print("Final fidelity:")
    print("IBM:  ", hellinger_fidelity(counts_expected, averga_counts_ibm))
    print("Stim: ", hellinger_fidelity(counts_expected, averga_counts_stim))
    print("Ours: ", hellinger_fidelity(counts_expected, averga_counts_ours))


def arch_data_output(backend_name):
    if backend_name == "vigo":
        backend = FakeVigo()
    elif backend_name == "mumbai":
        backend = FakeMumbai()
    elif backend_name == "guadalupe":
        backend = FakeGuadalupe()
    elif backend_name == "quito":
        backend = FakeQuito()
    elif backend_name == "nairobi":
        backend = FakeNairobi()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    topo = Topology.from_qiskit_backend(backend)

    print("Number of qubits: ", topo.num_qubits)
    print(topo.to_nx.edges)


class FakeJSONBackend(ConfigurableFakeBackend):
    def __init__(self, backend_name):
        with open("./backends_2023.json", "r") as f:
            backends = json.load(f)
        my_backend = None
        for backend in backends:
            if backend["name"] == backend_name:
                my_backend = backend
        if my_backend is None:
            raise ValueError(f"Unknown backend: {backend_name}")

        super().__init__(name=backend_name,
                         n_qubits=my_backend["qubits"],
                         coupling_map=my_backend["couplingMap"],
                         basis_gates=my_backend["basisGates"],
                         version=1.2,
                         single_qubit_gates=[g for g in my_backend["basisGates"] if
                                             g != "cx"],
                         )


if __name__ == "__main__":
    # open backends_2023.json

    # run_clifford_real_hardware(backend_name="ibmq_quito")
    # run_clifford_real_hardware(backend_name="ibm_nairobi")

    # analyze_real_hw(backend_name="ibmq_quito")
    # analyze_real_hw(backend_name="ibm_nairobi")

    # random_experiment(backend_name="complete", nr_input_gates=350, nr_steps=20)
    # random_experiment(backend_name="guadalupe", nr_input_gates=350, nr_steps=20)
    # random_experiment(backend_name="mumbai", nr_input_gates=700, nr_steps=40)
    # random_experiment(backend_name="ithaca", nr_input_gates=1800, nr_steps=100)
    random_experiment(backend_name="brisbane", nr_input_gates=7200, nr_steps=400)
    # random_experiment(backend_name="seattle", nr_input_gates=7200, nr_steps=400)

    # plot_experiment(name="random_quito")
    # plot_experiment(name="random_ithaca")
    # plot_experiment(name="random_complete")
    # plot_experiment(name="random_guadalupe")
    # plot_experiment(name="random_mumbai")
