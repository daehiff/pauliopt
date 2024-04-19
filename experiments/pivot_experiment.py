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

from qiskit.qasm2 import dumps

from clifford_experiment import duncan_et_al_synthesis, get_backend_and_df_name, maslov_et_al_compilation, our_compilation, our_compilation_heat, qiskit_compilation, qiskit_tableau_compilation, random_hscx_circuit, stim_compilation

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def pivot_experiment(backend_name="vigo", nr_input_gates=100, nr_steps=5, df_name="data/random"):
    backend, df_name = get_backend_and_df_name(backend_name, df_name=df_name)
    if backend not in ["complete", "line"]:
        num_qubits = backend.configuration().num_qubits
    else:
        num_qubits = int(backend_name.split("_")[1])

    print(df_name)
    print(num_qubits)
    df = pd.DataFrame(
        columns=["n_rep", "num_qubits", "n_gadgets", "method", "h", "s", "cx", "time", "depth", "2q_depth"])
    for n_gadgets in range(1, nr_input_gates, nr_steps):
        print(n_gadgets)
        for _ in range(20):
            circ = random_hscx_circuit(
                nr_qubits=num_qubits, nr_gates=n_gadgets)

            ########################################
            # Our clifford circuit
            ########################################
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "method": "ours"} | \
                our_compilation(circ, backend)
            df.loc[len(df)] = column
            df.to_csv(df_name)

            ########################################
            # Our clifford circuit
            ########################################
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "method": "ours_random"} | \
                our_compilation(circ, backend, choice_fn=random_chooser)
            df.loc[len(df)] = column
            df.to_csv(df_name)

            ########################################
            # Our clifford circuit w/ temp
            ########################################
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "method": "ours_temp"} | \
                our_compilation_heat(circ, backend, choice_fn=heat_chooser)
            df.loc[len(df)] = column
            df.to_csv(df_name)

            ########################################
            # Bravi et. al.
            ########################################
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "method": "Bravyi et al. (qiskit)"} | \
                qiskit_tableau_compilation(circ, backend)
            df.loc[len(df)] = column
            df.to_csv(df_name)

    df.to_csv(df_name)
    # print(circ_out)
    # print(circ_stim)


if __name__ == "__main__":
    # run_clifford_real_hardware(backend_name="ibmq_quito")
    # run_clifford_real_hardware(backend_name="ibm_nairobi")

    # analyze_real_hw(backend_name="ibmq_quito")
    # analyze_real_hw(backend_name="ibm_nairobi")
    df_name = "data/pivot/random"

    # pivot_experiment(backend_name="quito", nr_input_gates=200,
    #                  nr_steps=20, df_name=df_name)
    # pivot_experiment(backend_name="complete_5",
    #                  nr_input_gates=200, nr_steps=20, df_name=df_name)

    # pivot_experiment(backend_name="nairobi", nr_input_gates=300,
    #                  nr_steps=20, df_name=df_name)
    # pivot_experiment(backend_name="complete_7",
    #                  nr_input_gates=300, nr_steps=20, df_name=df_name)

    # pivot_experiment(backend_name="guadalupe",
    #                  nr_input_gates=400, nr_steps=20, df_name=df_name)
    # pivot_experiment(backend_name="complete_16",
    #                  nr_input_gates=400, nr_steps=20, df_name=df_name)

    # pivot_experiment(backend_name="mumbai", nr_input_gates=800,
    #                  nr_steps=40, df_name=df_name)
    # pivot_experiment(backend_name="complete_27",
    #                  nr_input_gates=800, nr_steps=40, df_name=df_name)
    #
    # pivot_experiment(backend_name="ithaca", nr_input_gates=2000,
    #                  nr_steps=100, df_name=df_name)
    # pivot_experiment(backend_name="complete_65",
    #                  nr_input_gates=2000, nr_steps=100, df_name=df_name)

    # pivot_experiment(backend_name="brisbane",
    #                  nr_input_gates=10000, nr_steps=400, df_name=df_name)
    # pivot_experiment(backend_name="complete_127",
    #                  nr_input_gates=10000, nr_steps=400, df_name=df_name)

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
