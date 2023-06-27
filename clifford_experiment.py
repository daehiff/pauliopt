import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import FakeVigo, FakeMumbai, FakeGuadalupe
from qiskit.quantum_info import Clifford, random_clifford

from pauliopt.pauli.clifford_tableau import CliffordTableau
from pauliopt.pauli.utils import apply_permutation
from pauliopt.topologies import Topology
from test import verify_equality


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


def qiskit_tableau_compilation_tableau(tableau: Clifford, backend):
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
    circ_out, perm = ct.to_cifford_circuit_arch_aware(topo, improvements=False)
    column = get_ops_count(circ_out)
    #assert verify_equality(circ, circ_out)
    print("Our: ", circ_out.count_ops())
    return column


def our_compilation_imp(circ: QuantumCircuit, backend):
    topo = Topology.from_qiskit_backend(backend)
    ct = CliffordTableau.from_circuit(circ)
    circ_out, perm = ct.to_cifford_circuit_arch_aware(topo, improvements=True)
    column = get_ops_count(circ_out)
    #assert verify_equality(circ, circ_out)
    print("Our Imp: ", circ_out.count_ops())
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
                      "method": "tableau"} | our_compilation(circ, backend)
            df.loc[len(df)] = column
            df.to_csv(df_name)

            ########################################
            # Our clifford circuit Improvements
            ########################################
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "method": "tableau_imp"} | our_compilation_imp(circ, backend)
            df.loc[len(df)] = column
            df.to_csv(df_name)

            # ########################################
            # # Qiskit compilation
            # ########################################
            # column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
            #           "method": "qiskit"} | qiskit_compilation(circ, backend)
            # df.loc[len(df)] = column
            # df.to_csv(df_name)

            ########################################
            # Qiskit tableau compilation
            ########################################
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "method": "qiskit_tableau"} \
                     | qiskit_compilation(circ, backend)
            df.loc[len(df)] = column
            df.to_csv(df_name)

    df.to_csv(df_name)
    # print(circ_out)
    # print(circ_stim)


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


if __name__ == "__main__":
    random_experiment(backend_name="vigo", nr_input_gates=70, nr_steps=5)
    # random_experiment(backend_name="guadalupe", nr_input_gates=250, nr_steps=10)
    # random_experiment(backend_name="mumbai", nr_input_gates=600, nr_steps=20)
    # plot_mumbai()
    plot_vigo()
    # plot_guadalupe()
