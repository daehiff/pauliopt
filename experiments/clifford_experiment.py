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

from tests.test_clifford_tableau import verify_equality

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_2q_depth(qc: QuantumCircuit):
    c = qiskit_to_tk(qc)
    return {"2q_depth": c.depth_2q()}


def get_depth(qc: QuantumCircuit):
    return {"depth": qc.depth()}


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
            target = np.random.choice(
                [i for i in range(nr_qubits) if i != control])
            qc.cx(control, target)
        elif gate_t == "CY":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice(
                [i for i in range(nr_qubits) if i != control])
            qc.cy(control, target)
        elif gate_t == "CZ":
            control = np.random.choice([i for i in range(nr_qubits)])
            target = np.random.choice(
                [i for i in range(nr_qubits) if i != control])
            qc.cz(control, target)
    return qc


def random_clifford_circuit(nr_qubits=4):
    clifford = qiskit.quantum_info.random_clifford(nr_qubits)
    return clifford


def get_ops_count(qc: QuantumCircuit):
    count = {
        "h": 0,
        "cx": 0,
        "s": 0,
        "depth": qc.depth(lambda inst: inst.operation.name == "cx"),
    }
    ops = qc.count_ops()
    if "cx" in ops.keys():
        count["cx"] += ops["cx"]
    if "h" in ops.keys():
        count["h"] += ops["h"]
    if "s" in ops.keys():
        count["s"] += ops["s"]
    return count


def _get_qubits_qiskit(qubits, qreg):
    """
    Helper method to read the qubit indices from the qiskit quantum register.
    """
    qubits_ = []
    for qubit in qubits:
        qubits_.append(qreg.index(qubit))

    return tuple(qubits_)


def circuit_to_stim_tableau(circ: QuantumCircuit):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    tableau = stim.Tableau(circ.num_qubits)
    cnot = stim.Tableau.from_named_gate("CX")
    had = stim.Tableau.from_named_gate("H")
    s = stim.Tableau.from_named_gate("S")

    qreg = circ.qregs[0]
    for op in circ:
        qubits = _get_qubits_qiskit(op.qubits, qreg)
        if op.operation.name == "h":
            tableau.append(had, [qubits[0]])
        elif op.operation.name == "s":
            tableau.append(s, [qubits[0]])
        elif op.operation.name == "cx":
            tableau.append(cnot, [qubits[0], qubits[1]])
        else:
            raise Exception("Unknown operation")
    return tableau


def parse_stim_to_qiskit(circ: stim.Circuit):
    qc = QuantumCircuit(circ.num_qubits)
    for gate in circ:
        if gate.name == "CX":
            targets = [target.value for target in gate.targets_copy()]
            targets = [(targets[i], targets[i + 1])
                       for i in range(0, len(targets), 2)]
            for ctrl, target in targets:
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
        circ_out = transpile(circ_out,
                             routing_method="sabre",
                             basis_gates=['s', 'h', 'cx'])
    elif backend == "line":
        circ_out = transpile(circ, coupling_map=[[i, i + 1] for i in
                                                 range(circ.num_qubits - 1)],
                             routing_method="sabre",
                             basis_gates=['s', 'h', 'cx'])
    else:
        circ_out = transpile(circ_out,
                             coupling_map=backend.coupling_map,
                             routing_method="sabre",
                             basis_gates=['s', 'h', 'cx'])
    act_time = time.time() - start
    # print("Qiskit Tableau: ", circ_out.count_ops(),
    #   "Time: ", time.time() - start)
    column = get_ops_count(circ_out) | {"time": act_time}
    column = column | get_depth(circ_out) | get_2q_depth(circ_out)

    return column


def qiskit_tableau_compilation(circ: QuantumCircuit, backend):
    tableau = Clifford.from_circuit(circ)
    circ_out = tableau.to_circuit()
    start = time.time()
    if backend == "complete":
        circ_out = transpile(
            circ_out, routing_method="sabre", basis_gates=["s", "h", "cx"]
        )
    elif backend == "line":
        circ_out = transpile(
            circ,
            coupling_map=[[i, i + 1] for i in range(circ.num_qubits - 1)],
            routing_method="sabre",
            basis_gates=["s", "h", "cx"],
        )
    else:
        circ_out = transpile(circ_out,
                             routing_method="sabre",
                             coupling_map=backend.coupling_map,
                             basis_gates=["cx", "h", "s"])
    act_time = time.time() - start
    # print("Qiskit Tableau: ", circ_out.count_ops(),
    #       "Time: ", time.time() - start)
    column = get_ops_count(circ_out) | {"time": act_time}
    column = column | get_depth(circ_out) | get_2q_depth(circ_out)

    return column


def qiskit_compilation(circ: QuantumCircuit, backend):
    start = time.time()
    if backend == "complete":
        circ_out = transpile(circ, routing_method="sabre",
                             basis_gates=["s", "h", "cx"])
    elif backend == "line":
        circ_out = transpile(
            circ,
            coupling_map=[[i, i + 1] for i in range(circ.num_qubits - 1)],
            routing_method="sabre",
            basis_gates=["h", "s", "cx"],
        )
    else:
        circ_out = transpile(circ, coupling_map=backend.coupling_map,
                             basis_gates=['s', 'h', 'cx'])
    act_time = time.time() - start
    column = get_ops_count(circ_out) | {"time": act_time}
    column = column | get_depth(circ_out) | get_2q_depth(circ_out)
    # print("Qiskit: ", circ_out.count_ops(), "Time: ", act_time)
    return column


def optimal_compilation(clifford: qiskit.quantum_info.Clifford, backend):
    start = time.time()

    circ, _ = qmap.synthesize_clifford(clifford,
                                       # use_maxsat=True,
                                       dump_intermediate_results=True,
                                       include_destabilizers=True)
    start = time.time()
    if backend == "complete":
        circ_out = transpile(circ, routing_method="sabre",
                             basis_gates=["s", "h", "cx"])
    elif backend == "line":
        circ_out = transpile(
            circ,
            routing_method="sabre",
            coupling_map=[[i, i + 1] for i in range(circ.num_qubits - 1)],
            basis_gates=["s", "h", "cx"],
        )
    else:
        circ_out = transpile(circ, coupling_map=backend.coupling_map,
                             routing_method="sabre",
                             basis_gates=['s', 'h', 'cx'])
    act_time = time.time() - start
    column = get_ops_count(circ_out) | {
        "time": act_time, "depth": circ_out.depth()}
    # print("Optimal: ", circ_out.count_ops(), "Time: ", act_time)
    return column


def our_compilation(circ: QuantumCircuit, backend, choice_fn=min):
    if backend == "complete":
        topo = Topology.complete(circ.num_qubits)
    elif backend == "line":
        topo = Topology.line(circ.num_qubits)
    else:
        topo = Topology.from_qiskit_backend(backend)
    ct = CliffordTableau.from_circuit(circ)
    start = time.time()
    circ_out, _ = ct.to_cifford_circuit_arch_aware(
        topo, include_swaps=True, choice_fn=choice_fn)
    act_time = time.time() - start
    circ_out = circ_out.to_qiskit()
    column = get_ops_count(circ_out) | {"time": act_time}
    column = column | get_depth(circ_out) | get_2q_depth(circ_out)

    assert verify_equality(circ, circ_out)
    # print("Our: ", circ_out.count_ops(), "Time: ", act_time)
    return column


def our_compilation_heat(circ: QuantumCircuit, backend, choice_fn=min):
    if backend == "complete":
        topo = Topology.complete(circ.num_qubits)
    elif backend == "line":
        topo = Topology.line(circ.num_qubits)
    else:
        topo = Topology.from_qiskit_backend(backend)
    ct = CliffordTableau.from_circuit(circ)
    start = time.time()
    best_circ = None
    best_cx_count = math.inf
    for _ in range(40):
        circ_out, _ = ct.to_cifford_circuit_arch_aware(
            topo, include_swaps=True, choice_fn=choice_fn)
        circ_out = circ_out.to_qiskit()
        column = get_ops_count(circ_out)
        if column["cx"] < best_cx_count:
            best_cx_count = column["cx"]
            best_circ = circ_out
    act_time = time.time() - start
    column = get_ops_count(best_circ) | {"time": act_time}
    column = column | get_depth(best_circ) | get_2q_depth(best_circ)
    assert verify_equality(circ, circ_out)
    # print("Our: ", circ_out.count_ops(), "Time: ", act_time)
    return column


def maslov_et_al_compilation(circ: QuantumCircuit, backend):
    tableau = Clifford.from_circuit(circ)
    start = time.time()
    circ_out = synth_clifford_depth_lnn(tableau)
    if backend == "complete":
        circ_out = transpile(
            circ_out, routing_method="sabre", basis_gates=["s", "h", "cx"]
        )
    elif backend == "line":
        circ_out = transpile(
            circ,
            coupling_map=[[i, i + 1] for i in range(circ.num_qubits - 1)],
            routing_method="sabre",
            basis_gates=["s", "h", "cx"],
        )
    else:
        circ_out = transpile(
            circ_out,
            routing_method="sabre",
            coupling_map=backend.coupling_map,
            basis_gates=["cx", "h", "s"],
        )
    act_time = time.time() - start
    # print("Masolv et. al: ", circ_out.count_ops(), "Time: ", time.time() - start)
    column = get_ops_count(circ_out) | {"time": act_time}
    column = column | get_depth(circ_out) | get_2q_depth(circ_out)

    return column


def rebase_pyzx(circ: QuantumCircuit):
    circ_out = QuantumCircuit(circ.num_qubits)
    qreg = circ.qregs[0]
    for instruction in circ:
        if instruction.operation.name == "rz":
            qubits = _get_qubits_qiskit(instruction.qubits, qreg)
            if (
                instruction.operation.params[0] == np.pi / 2
                or instruction.operation.params[0] == -3 * np.pi / 2
            ):
                circ_out.s(qubits[0])
            elif (
                instruction.operation.params[0] == -np.pi / 2
                or instruction.operation.params[0] == 3 * np.pi / 2
            ):
                circ_out.sdg(qubits[0])
            elif (
                instruction.operation.params[0] == np.pi
                or instruction.operation.params[0] == -np.pi
            ):
                circ_out.z(qubits[0])
            else:
                print(instruction)
                raise Exception(
                    "Unknown rz: ", instruction.operation.decompositions)
        else:
            circ_out.append(instruction)
    return circ_out


def duncan_et_al_synthesis(circ: QuantumCircuit, backend):
    start = time.time()
    qasm_str = dumps(circ)
    circ_zx = zx.Circuit.from_qasm(qasm_str)
    circ_graph = circ_zx.to_graph()
    zx.clifford_simp(circ_graph)

    circ_zx = zx.extract_circuit(circ_graph)

    circ_out = QuantumCircuit.from_qasm_str(circ_zx.to_qasm())
    # small rebase since qiskit cannot handle r_z clifford rotations
    circ_out_ = rebase_pyzx(circ_out)
    # assert qiskit.quantum_info.Operator.from_circuit(circ_out).equiv(
    #     qiskit.quantum_info.Operator.from_circuit(circ_out_)
    # )
    circ_out = circ_out_
    if backend == "complete":
        circ_out = transpile(
            circ_out, routing_method="sabre", basis_gates=["s", "h", "cx"]
        )
    elif backend == "line":
        circ_out = transpile(
            circ,
            coupling_map=[[i, i + 1] for i in range(circ.num_qubits - 1)],
            routing_method="sabre",
            basis_gates=["s", "h", "cx"],
        )
    else:
        circ_out = transpile(
            circ_out,
            routing_method="sabre",
            coupling_map=backend.coupling_map,
            basis_gates=["cx", "h", "s"],
        )
    act_time = time.time() - start
    # print("Duncan et. al: ", circ_out.count_ops(), "Time: ", time.time() - start)
    column = get_ops_count(circ_out) | {"time": act_time}
    column = column | get_depth(circ_out) | get_2q_depth(circ_out)
    return column


def our_compilation_tableau(tab: Clifford, backend, num_qubits):
    if backend == "complete":
        topo = Topology.complete(num_qubits)
    elif backend == "line":
        topo = Topology.line(num_qubits)
    else:
        topo = Topology.from_qiskit_backend(backend)
    ct = CliffordTableau.from_qiskit(tab)
    start = time.time()
    circ_out, perm = ct.to_cifford_circuit_arch_aware(topo, include_swaps=True)
    act_time = time.time() - start
    circ_out = circ_out.to_qiskit()
    column = get_ops_count(circ_out)
    # assert verify_equality(tab.to_circuit(), apply_permutation(circ_out, perm))
    # print("Our: ", circ_out.count_ops(), "Time: ", act_time)
    return column


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


def random_experiment_complete(backend_name="vigo"):
    backend, df_name = get_backend_and_df_name(
        backend_name, df_name="data/random_converged"
    )
    if backend not in ["complete", "line"]:
        num_qubits = backend.configuration().num_qubits
    else:
        num_qubits = int(backend_name.split("_")[1])
    print(num_qubits)
    print(df_name)
    df = pd.DataFrame(
        columns=["n_rep", "num_qubits", "method", "h", "s", "cx", "depth"]
    )

    for _ in range(20):
        clifford = random_clifford_circuit(nr_qubits=num_qubits)
        column = {
            "n_rep": _,
            "num_qubits": num_qubits,
            "method": "ours",
        } | our_compilation_tableau(clifford, backend, num_qubits)
        df.loc[len(df)] = column
        df.to_csv(df_name)

        column = {
            "n_rep": _,
            "num_qubits": num_qubits,
            "method": "optimal",
        } | optimal_compilation(clifford, backend)
        df.loc[len(df)] = column
        df.to_csv(df_name)
    df.to_csv(df_name)

    print(df.groupby("method").mean())


def random_experiment(backend_name="vigo", nr_input_gates=100, nr_steps=5, df_name="data/random"):
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
            # column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
            #           "method": "ours_random"} | \
            #     our_compilation(circ, backend, choice_fn=random_chooser)
            # df.loc[len(df)] = column
            # df.to_csv(df_name)

            ########################################
            # Our clifford circuit w/ temp
            ########################################
            # column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
            #           "method": "ours_temp"} | \
            #     our_compilation_heat(circ, backend, choice_fn=heat_chooser)
            # df.loc[len(df)] = column
            # df.to_csv(df_name)

            # ########################################
            # # Qiskit compilation
            # ########################################
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "method": "qiskit transpile"} | \
                qiskit_compilation(circ, backend)
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

            ########################################
            # Stim compilation
            ########################################
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "method": "Ewout van den Berg (stim)"} | \
                stim_compilation(circ, backend)
            df.loc[len(df)] = column
            df.to_csv(df_name)

            ########################################
            # Stim compilation
            ########################################
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "method": "Duncan et al (pyzx)"} | \
                duncan_et_al_synthesis(circ, backend)
            df.loc[len(df)] = column
            df.to_csv(df_name)

            ########################################
            # Stim compilation
            ########################################
            column = {"n_rep": _, "num_qubits": num_qubits, "n_gadgets": n_gadgets,
                      "method": "Maslov et al"} | \
                maslov_et_al_compilation(circ, backend)
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


# def run_clifford_experiment(exp_number=0, backend_name="ibm_perth"):
#     base_path = f"data/clifford_experiment_real_hardware/{backend_name}/{exp_number}"
#     if not os.path.exists(base_path):
#         os.makedirs(base_path)
#
#     # read token from env IBM_TOKEN
#     token = os.environ.get("IBM_TOKEN")
#     assert token
#     # provider = IBMQ.enable_account(token)
#     provider = IBMProvider(token=token, instance="ibm-q-research-2/tu-munich-2/main")
#     backend = provider.get_backend(backend_name)
#     if backend_name == "ibm_perth" or backend_name == "ibm_nairobi":
#         nr_gates = 30
#     else:
#         nr_gates = 25
#     # Get the provider and choose the backend
#     clifford_circ = random_hscx_circuit(nr_gates=nr_gates,
#                                         nr_qubits=backend.configuration().num_qubits)
#     clifford = Clifford.from_circuit(clifford_circ)
#     ct = CliffordTableau(tableau=clifford.symplectic_matrix, signs=clifford.phase)
#     with open(f"{base_path}/clifford.json", "w") as f:
#         json.dump(clifford.to_dict(), f)
#
#     backend_simulator = Aer.get_backend('statevector_simulator')
#     circ_simulated = clifford.to_circuit()
#     circ_simulated_ = QuantumCircuit(circ_simulated.num_qubits,
#                                      name=f"clifford_simulated_{exp_number}")
#     circ_simulated_.compose(circ_simulated, inplace=True)
#     circ_simulated_.barrier()
#     circ_simulated_.compose(circ_simulated.inverse(), inplace=True)
#     circ_simulated_.measure_all()
#     circ_simulated = circ_simulated_
#     circ_simulated.qasm(filename=f"{base_path}/clifford_simulated.qasm")
#     job = execute(circ_simulated, backend_simulator, shots=16000)
#     result = job.result()
#     counts_simulated = result.get_counts()
#     print(counts_simulated)
#
#     ######################
#     # Stim execution
#     ######################
#
#     tableau = circuit_to_stim_tableau(clifford_circ)
#     circ_stim = parse_stim_to_qiskit(tableau.to_circuit(method="elimination"))
#     circ_stim = transpile(circ_stim, backend)
#
#     circ_stim_ = QuantumCircuit(circ_stim.num_qubits,
#                                 name=f"clifford_stim_{exp_number}")
#     circ_stim_.compose(circ_stim, inplace=True)
#     circ_stim_.barrier()
#     circ_stim_.compose(circ_stim.inverse(), inplace=True)
#     circ_stim_.measure_all()
#
#     job_stim = execute(circ_stim_, backend, shots=16000)
#
#     print("Stim: ", circ_stim_.count_ops()["cx"])
#
#     while job_stim.status() != JobStatus.DONE:
#         time.sleep(1)
#     result_stim = job_stim.result()
#     with open(f"{base_path}/result_stim.json", "w") as f:
#         json.dump(result_stim.to_dict(), f, default=json_serial)
#
#     circ_stim_.qasm(filename=f"{base_path}/stim.qasm")
#
#     ######################
#     # Our execution of the quantum circuit
#     ######################
#
#     topo = Topology.from_qiskit_backend(backend)
#     circ_ours, perm = ct.to_cifford_circuit_arch_aware(topo)
#     circ_ours = apply_permutation(circ_ours.to_qiskit(), perm)
#
#     circ_ours_ = QuantumCircuit(circ_ours.num_qubits,
#                                 name=f"clifford_synth_{exp_number}")
#     circ_ours_.compose(circ_ours, inplace=True)
#     circ_ours_.barrier()
#     circ_ours_.compose(circ_ours.inverse(), inplace=True)
#     circ_ours_.measure_all()
#     circ_ours = circ_ours_
#     circ_ours.qasm(filename=f"{base_path}/ours.qasm")
#
#     job_ours = execute(circ_ours, backend, shots=16000)
#
#     ######################
#     # IBM execution of the quantum circuit
#     ######################
#     circ = clifford.to_circuit()
#
#     circ = transpile(circ, backend)
#
#     circ_ = QuantumCircuit(circ_ours.num_qubits,
#                            name=f"ibm_synth_{exp_number}")
#     circ_.compose(circ, inplace=True)
#     circ_.barrier()
#     circ_.compose(circ.inverse(), inplace=True)
#     circ_.measure_all()
#     circ = circ_
#     circ.qasm(filename=f"{base_path}/ibm.qasm")
#     job_ibm = execute(circ, backend, shots=16000)
#
#     print("Ours: ", circ_ours.count_ops()["cx"])
#     print("IBM: ", circ.count_ops()["cx"])
#
#     while job_ibm.status() != JobStatus.DONE:
#         time.sleep(1)
#
#     result_ibm = job_ibm.result()
#     with open(f"{base_path}/result_ibm.json", "w") as f:
#         json.dump(result_ibm.to_dict(), f, default=json_serial)
#
#     while job_ours.status() != JobStatus.DONE:
#         time.sleep(1)
#     result_ours = job_ours.result()
#     with open(f"{base_path}/result_ours.json", "w") as f:
#         json.dump(result_ours.to_dict(), f, default=json_serial)
#
#     count_ours = result_ours.get_counts()
#     count_ours = apply_permutation_measurements(count_ours, perm)
#     count_ibm = result_ibm.get_counts()
#     count_stim = result_stim.get_counts()
#
#     fidelity_ours = hellinger_fidelity(count_ours, counts_simulated)
#     fidelity_ibm = hellinger_fidelity(count_ibm, counts_simulated)
#     fidelity_stim = hellinger_fidelity(count_stim, counts_simulated)
#
#     print("Ours: ", fidelity_ours)
#     print("IBM: ", fidelity_ibm)
#     print("Stim: ", fidelity_stim)
#
#     col = {
#         "cx_ours": circ_ours.count_ops()["cx"],
#         "cx_ibm": circ.count_ops()["cx"],
#         "fidelity_ours": fidelity_ours,
#         "fidelity_ibm": fidelity_ibm,
#         "time_ours": result_ours.time_taken,
#         "time_ibm": result_ibm.time_taken
#     }
#     return pd.DataFrame(col, index=[0])
#


def plot_experiment(name="random_guadalupe", v_line_cx=None):
    plt.rcParams.update(
        {"text.usetex": True, "font.family": "sans-serif", "font.size": 11}
    )
    sns.set_palette(sns.color_palette("colorblind"))

    df = pd.read_csv(f"data/{name}.csv")
    # df = df[df["method"] != "Ewout van den Berg (stim)"]
    # df["method"] = df["method"].replace(
    #     {
    #         "qiskit transpile": "qiskit",
    #         "ours": "tableau (ours)",
    #         "Bravyi et al. (qiskit)": "qiskit_tableau",
    #     }
    # )

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
    # add a vertical line for v_line_cx if not none
    if v_line_cx is not None:
        plt.axhline(y=v_line_cx, color="black", linestyle="--")
    plt.savefig(f"data/{name}_cx.pdf")
    plt.show()


# def run_clifford_real_hardware(backend_name="ibm_perth"):
#     experiment_numbers = [i for i in range(0, 20)]
#     run_clifford_experiment_ = functools.partial(run_clifford_experiment,
#                                                  backend_name=backend_name)
#     # create a threadpool and run run_clifford_experiment
#     with ThreadPoolExecutor(max_workers=20) as executor:
#         # Submit tasks to the thread pool
#
#         results = executor.map(run_clifford_experiment_, experiment_numbers)
#     df = pd.concat(results, ignore_index=True)
#     print(df)
#     df.to_csv("data/clifford_experiment_real_hardware/results.csv")


def analyze_real_hw(backend_name="ibm_nairobi"):
    cumult_counts_ibm = {}
    cumult_counts_ours = {}
    cumult_counts_stim = {}
    num_qubits = 1

    df_fid = pd.DataFrame(
        columns=["fid_our", "fid_ibm", "fid_stim",
                 "time_our", "time_ibm", "time_stim"]
    )
    df = pd.DataFrame(columns=["method", "h", "s",
                      "cx", "fid", "execution_time"])
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
            if "h" in circ_ours.count_ops()
            else 0,
            "s": circ_ours.count_ops()["s"] / 2.0
            if "s" in circ_ours.count_ops()
            else 0,
            "cx": circ_ours.count_ops()["cx"] / 2.0
            if "cx" in circ_ours.count_ops()
            else 0,
            "fid": fid_ours,
            "execution_time": result_ours.time_taken,
        }
        df.loc[len(df)] = column

        column = {
            "method": "stim",
            "h": circ_stim.count_ops()["h"] / 2.0
            if "h" in circ_stim.count_ops()
            else 0,
            "s": circ_stim.count_ops()["s"] / 2.0
            if "s" in circ_stim.count_ops()
            else 0,
            "cx": circ_stim.count_ops()["cx"] / 2.0
            if "cx" in circ_stim.count_ops()
            else 0,
            "fid": fid_stim,
            "execution_time": result_stim.time_taken,
        }
        df.loc[len(df)] = column

        column = {
            "fid_our": fid_ours,
            "fid_ibm": fid_ibm,
            "fid_stim": fid_stim,
            "time_our": result_ours.time_taken,
            "time_ibm": result_ibm.time_taken,
            "time_stim": result_stim.time_taken,
        }
        df_fid.loc[len(df_fid)] = column

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

    df.to_csv(
        f"data/clifford_experiment_real_hardware/{backend_name}/analysis.csv")

    for name in ["ours", "stim", "ibm"]:
        print(
            f"Mean execution time {name}: {df[df['method'] == name]['execution_time'].mean()}"
        )

    print("=====================================")
    print("Correlation Matrix:")

    df_ = df[df["method"] == "ours"]
    correlation_matrix = df_[["h", "s", "cx", "fid"]].corr()
    print("Correlation Matrix CNOTs:")
    print(correlation_matrix)
    print(df.groupby("method").mean())

    print("=====================================")
    print("Individual Fidelities:")
    print(df_fid)
    print(df_fid.mean().T.to_latex())
    print("=====================================")

    lables = []
    values = []
    for key, value in cumult_counts_ours.items():
        lables.append([key] * len(value))
        values.append(value)
    lables = list(np.hstack(lables))
    values = list(np.hstack(values))
    cat = ["ours"] * len(lables)
    df_our = pd.DataFrame(
        {"approach": cat, "lables": lables, "values": values})

    lables = []
    values = []
    for key, value in cumult_counts_ibm.items():
        lables.append([key] * len(value))
        values.append(value)
    lables = list(np.hstack(lables))
    values = list(np.hstack(values))
    cat = ["Bravyi et al. (qiskit)"] * len(lables)
    df_ibm = pd.DataFrame(
        {"approach": cat, "lables": lables, "values": values})

    lables = []
    values = []
    for key, value in cumult_counts_stim.items():
        lables.append([key] * len(value))
        values.append(value)

    lables = list(np.hstack(lables))
    values = list(np.hstack(values))
    cat = ["Ewout van den Berg (stim)"] * len(lables)
    df_stim = pd.DataFrame(
        {"approach": cat, "lables": lables, "values": values})
    df = pd.concat([df_our, df_ibm, df_stim], ignore_index=True)

    # seaborn barplot rotate the x ticks labels by 90 degrees
    plt.rcParams.update(
        {"text.usetex": True, "font.family": "sans-serif", "font.size": 11}
    )
    sns.set_palette(sns.color_palette("colorblind"))
    sns.barplot(data=df, x="lables", y="values", hue="approach", errwidth=0.1)
    plt.title("Average counts for 20 random Clifford circuits")
    plt.xticks(rotation=90)
    plt.xlabel("Measured State")
    plt.ylabel("Counts")
    plt.tight_layout()
    plt.savefig("data/clifford_experiment_real_hardware/counts_clifford.pdf")
    plt.show()


class FakeJSONBackend(FakeBackend):
    def __init__(self, backend_name):
        with open("./backends_2023.json", "r") as f:
            backends = json.load(f)
        my_backend = None
        for backend in backends:
            if backend["name"] == backend_name:
                my_backend = backend
        if my_backend is None:
            raise ValueError(f"Unknown backend: {backend_name}")

        config = BackendConfiguration(
            backend_name=backend_name,
            backend_version="0.0",
            n_qubits=my_backend["qubits"],
            basis_gates=my_backend["basisGates"],
            gates=[GateConfig(name="cx", parameters=[], qasm_def="cx")],
            local=True,
            simulator=False,
            conditional=False,
            open_pulse=False,
            memory=False,
            max_shots=2048,
            coupling_map=my_backend["couplingMap"],
        )

        super().__init__(config)

        self.coupling_map = my_backend["couplingMap"]


def estimate_routing_overhead(arch_name, n_qubits, conv_gadgets):
    print(
        f"Estimating routing overhead for {arch_name} with {n_qubits} qubits")

    for method in ["Bravyi et al. (qiskit)", "ours", "Ewout van den Berg (stim)"]:
        df = pd.read_csv(f"data/random_complete_{n_qubits}.csv")
        df_ = df[df["method"] == method]
        df_ = df_[df_["n_gadgets"] > conv_gadgets]

        best_cx_complete = np.mean(df_["cx"])

        df = pd.read_csv(f"data/random_{arch_name}.csv")
        df = df[df["method"] == method]
        df = df[df["n_gadgets"] > 75]
        best_cx_quito = np.mean(df["cx"])
        print(
            f"{method}: {100.0 * (best_cx_quito - best_cx_complete) / best_cx_quito:.2f} %"
        )


def get_complete_cx_count():
    for n_qubits, conv_gadgets in [
        (5, 75),
        (7, 110),
        (16, 250),
        (27, 500),
        (65, 1250),
        (127, 3000),
    ]:
        print("n_qubits: ", n_qubits)
        bound_l = np.round(n_qubits**2 / np.log2(n_qubits))
        bound_u = np.round(n_qubits**2)
        print("Bound_l: ", bound_l)
        print("Bound_u: ", bound_u)
        df = pd.read_csv(f"data/random_complete_{n_qubits}.csv")

        df["method"] = pd.Categorical(
            df["method"],
            categories=[
                "ours", "Bravyi et al. (qiskit)", "Ewout van den Berg (stim)"],
            ordered=True,
        )
        df = df.sort_values(by="method")
        df_ = df[df["n_gadgets"] > conv_gadgets]
        print("Complete: ")
        print(df_.groupby("method").mean().round()["cx"])
        print()
        print("Bound_l: ")
        print((df_.groupby("method").mean().round()["cx"] / bound_l).round(2))
        print()
        print("Bound_u: ")
        print((df_.groupby("method").mean().round()["cx"] / bound_u).round(2))


if __name__ == "__main__":
    # run_clifford_real_hardware(backend_name="ibmq_quito")
    # run_clifford_real_hardware(backend_name="ibm_nairobi")

    # analyze_real_hw(backend_name="ibmq_quito")
    # analyze_real_hw(backend_name="ibm_nairobi")
    df_name = "data/new_methods/random"

    # random_experiment(backend_name="quito", nr_input_gates=200,
    #                   nr_steps=20, df_name=df_name)
    # random_experiment(backend_name="complete_5",
    #                   nr_input_gates=200, nr_steps=20, df_name=df_name)

    # random_experiment(backend_name="nairobi",
    #                   nr_input_gates=300, nr_steps=20, df_name=df_name)
    # random_experiment(backend_name="complete_7",
    #                   nr_input_gates=300, nr_steps=20, df_name=df_name)

    # random_experiment(backend_name="guadalupe",
    #                   nr_input_gates=400, nr_steps=20, df_name=df_name)
    # random_experiment(backend_name="complete_16",
    #                   nr_input_gates=400, nr_steps=20, df_name=df_name)

    # random_experiment(backend_name="mumbai", nr_input_gates=800,
    #                   nr_steps=40, df_name=df_name)
    # random_experiment(backend_name="complete_27",
    #                   nr_input_gates=800, nr_steps=40, df_name=df_name)

    # random_experiment(backend_name="ithaca",
    #                   nr_input_gates=2000, nr_steps=100, df_name=df_name)
    # random_experiment(backend_name="complete_65",
    #                   nr_input_gates=2000, nr_steps=100, df_name=df_name)

    random_experiment(backend_name="brisbane",
                      nr_input_gates=10000, nr_steps=400, df_name=df_name)
    random_experiment(backend_name="complete_127",
                      nr_input_gates=10000, nr_steps=400, df_name=df_name)

    # random_experiment_complete(backend_name="line_3")
    # random_experiment_complete(backend_name="line_4")
    # random_experiment_complete(backend_name="line_5")
    #
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
