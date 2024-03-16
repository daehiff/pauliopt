from clifford_experiment import duncan_et_al_synthesis, get_backend_and_df_name, maslov_et_al_compilation, our_compilation, our_compilation_heat, qiskit_compilation, qiskit_tableau_compilation, random_hscx_circuit, stim_compilation
from pytket.extensions.qiskit import qiskit_to_tk
from choice_fn import heat_chooser, random_chooser
import pyzx as zx
from pauliopt.topologies import Topology
from pauliopt.pauli.clifford_tableau import CliffordTableau
from qiskit.synthesis import (
    synth_clifford_depth_lnn,
    synth_clifford_bm,
    synth_clifford_greedy,
)
from qiskit.result import Result
from qiskit.quantum_info import Clifford, hellinger_fidelity
from qiskit.providers.models import BackendConfiguration, GateConfig
from qiskit.providers.fake_provider import FakeBackend
from qiskit import QuantumCircuit, transpile
from mqt import qmap
import stim
import seaborn as sns
import qiskit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
import functools
import json
import math
import os
import time
import warnings

from qiskit.qasm2 import dump
from pathlib import Path


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def gen_experimental_dataset(backend_name="vigo", nr_input_gates=100, nr_steps=5, df_name="data/random", seed=None):
    backend, output_csv = get_backend_and_df_name(
        backend_name, df_name=df_name)
    if backend not in ["complete", "line"]:
        num_qubits = backend.configuration().num_qubits
    else:
        num_qubits = int(backend_name.split("_")[1])

    # print(df_name)
    # print(num_qubits)
    # df = pd.DataFrame(
    #     columns=["n_rep", "num_qubits", "n_gadgets", "method", "h", "s", "cx", "time", "depth", "2q_depth"])
    folder_name = os.path.join(df_name, backend_name)
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    if seed:
        np.random.seed(seed)
    else:
        np.random.seed()
    for n_gadgets in range(1, nr_input_gates, nr_steps):
        for i in range(20):
            circ = random_hscx_circuit(
                nr_qubits=num_qubits, nr_gates=n_gadgets)
            circ_name = os.path.join(folder_name, backend_name + "_" + str(n_gadgets).zfill(4) +
                                     "_" + str(i).zfill(2) + ".qasm")
            with open(circ_name, mode="w") as f:
                dump(circ, f)

    # print(circ_out)
    # print(circ_stim)


if __name__ == "__main__":
    # run_clifford_real_hardware(backend_name="ibmq_quito")
    # run_clifford_real_hardware(backend_name="ibm_nairobi")

    # analyze_real_hw(backend_name="ibmq_quito")
    # analyze_real_hw(backend_name="ibm_nairobi")
    SEED = 42
    df_name = "datasets/clifford_experimental_dataset/"

    gen_experimental_dataset(backend_name="quito", nr_input_gates=200,
                             nr_steps=20, df_name=df_name, seed=SEED)
    gen_experimental_dataset(backend_name="complete_5",
                             nr_input_gates=200, nr_steps=20, df_name=df_name, seed=SEED)

    gen_experimental_dataset(backend_name="nairobi", nr_input_gates=300,
                             nr_steps=20, df_name=df_name, seed=SEED)
    gen_experimental_dataset(backend_name="complete_7",
                             nr_input_gates=300, nr_steps=20, df_name=df_name, seed=SEED)

    gen_experimental_dataset(backend_name="guadalupe",
                             nr_input_gates=400, nr_steps=20, df_name=df_name, seed=SEED)
    gen_experimental_dataset(backend_name="complete_16",
                             nr_input_gates=400, nr_steps=20, df_name=df_name, seed=SEED)

    gen_experimental_dataset(backend_name="mumbai", nr_input_gates=800,
                             nr_steps=40, df_name=df_name, seed=SEED)
    gen_experimental_dataset(backend_name="complete_27",
                             nr_input_gates=800, nr_steps=40, df_name=df_name, seed=SEED)
    #
    gen_experimental_dataset(backend_name="ithaca", nr_input_gates=2000,
                             nr_steps=100, df_name=df_name, seed=SEED)
    gen_experimental_dataset(backend_name="complete_65",
                             nr_input_gates=2000, nr_steps=100, df_name=df_name, seed=SEED)

    gen_experimental_dataset(backend_name="brisbane",
                             nr_input_gates=10000, nr_steps=400, df_name=df_name, seed=SEED)
    gen_experimental_dataset(backend_name="complete_127",
                             nr_input_gates=10000, nr_steps=400, df_name=df_name, seed=SEED)

    # plot_experiment(name="random_line_3", v_line_cx=None)

    # df = pd.read_csv(f"data/random_complete_5.csv")
    # df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    # df_ = df_[df_["n_gadgets"] > 75]
    # v_line = np.mean(df_["cx"])

    # plot_experiment(name="random_quito", v_line_cx=v_line)
    # plot_experiment(name="random_complete_5", v_line_cx=v_line)
    #
    #
    # df = pd.read_csv(f"data/random_complete_7.csv")
    # df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    # df_ = df_[df_["n_gadgets"] > 110]
    # v_line = np.mean(df_["cx"])
    #
    # plot_experiment(name="random_nairobi", v_line_cx=v_line)
    # plot_experiment(name="random_complete_7", v_line_cx=v_line)
    #
    # df = pd.read_csv(f"data/random_complete_16.csv")
    # df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    # df_ = df_[df_["n_gadgets"] > 250]
    # v_line = np.mean(df_["cx"])
    # #
    # plot_experiment(name="random_guadalupe", v_line_cx=v_line)
    # plot_experiment(name="random_complete_16", v_line_cx=v_line)
    #
    # df = pd.read_csv(f"data/random_complete_27.csv")
    # df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    # df_ = df_[df_["n_gadgets"] > 500]
    # v_line = np.mean(df_["cx"])
    #
    # plot_experiment(name="random_mumbai", v_line_cx=None)
    # plot_experiment(name="random_complete_27", v_line_cx=v_line)
    #
    # df = pd.read_csv(f"data/random_complete_65.csv")
    # df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    # df_ = df_[df_["n_gadgets"] > 1250]
    # v_line = np.mean(df_["cx"])
    #
    # plot_experiment(name="random_ithaca", v_line_cx=v_line)
    # plot_experiment(name="random_complete_65", v_line_cx=v_line)

    # df = pd.read_csv(f"data/random_complete_127.csv")
    # df_ = df[df["method"] == "Bravyi et al. (qiskit)"]
    # df_ = df_[df_["n_gadgets"] > 3000]
    # v_line = np.mean(df_["cx"])
    #
    # plot_experiment(name="random_brisbane", v_line_cx=v_line)
    # plot_experiment(name="random_complete_127", v_line_cx=v_line)

    # estimate_routing_overhead("quito", 5, 75)
    # estimate_routing_overhead("nairobi", 7, 110)
    # estimate_routing_overhead("guadalupe", 16, 250)
    # estimate_routing_overhead("mumbai", 27, 500)
    # estimate_routing_overhead("ithaca", 65, 1250)
    # estimate_routing_overhead("brisbane", 127, 3000)

    # get_complete_cx_count()
