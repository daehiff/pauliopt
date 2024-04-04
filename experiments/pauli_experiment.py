import itertools
import logging
import os, sys
import pickle
import shutil
from numbers import Number
from datetime import datetime

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
from pytket._tket.transform import Transform, PauliSynthStrat
from pytket.circuit import CXConfigType

from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.extensions.qiskit.backends import aer
from pytket.utils import gen_term_sequence_circuit, QubitPauliOperator
from qiskit import QuantumCircuit, transpile
from qiskit.providers.models import BackendConfiguration, GateConfig
from qiskit.providers.fake_provider import FakeBackend
from qiskit.quantum_info import process_fidelity
from sympy.core.symbol import Symbol

from paulihedral import synthesis_SC
from paulihedral.arch import dijkstra, pGraph
from paulihedral.benchmark.mypauli import pauliString
from paulihedral.parallel_bl import depth_oriented_scheduling
from pauliopt.pauli.pauli_gadget import PPhase
from pauliopt.pauli.pauli_polynomial import *
from pauliopt.pauli.synthesis import PauliSynthesizer, SynthMethod
from pauliopt.utils import pi, AngleVar
from multiprocessing import Pool, Manager, Queue
from itertools import product, repeat
import json
import networkx as nx


def get_2q_depth(qc: QuantumCircuit):
    return qc.depth(lambda inst: inst.operation.name == "cx")


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


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


def generate_random_z_polynomial(
    num_qubits: int, num_gadgets: int, min_legs=None, max_legs=None, allowed_angels=None
):
    if min_legs is None:
        min_legs = 2
    if max_legs is None:
        max_legs = num_qubits
    if allowed_angels is None:
        allowed_angels = [2 * pi, pi, pi / 2, pi / 4, pi / 8, pi / 16, pi / 32, pi / 64]
    allowed_legs = [Z]
    pp = PauliPolynomial(num_qubits)
    for _ in range(num_gadgets):
        pp >>= create_random_phase_gadget(
            num_qubits, min_legs, max_legs, allowed_angels, allowed_legs=allowed_legs
        )
    return pp


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
            # elif isinstance(coeff, Number):
            #     pp >>= PPhase(Angle(float(coeff))) @ paulis
            elif isinstance(coeff, Symbol):
                pp >>= PPhase(AngleVar(coeff.name)) @ paulis
            else:
                raise Exception("Unknown type")
        else:
            pp.global_phase += coeff
        # idx += 1
    return pp


def synth_tket(operator, topo: Topology, method: PauliSynthStrat, n_qubits: int = None):
    if n_qubits is None:
        n_qubits = topo.num_qubits

    initial_circ = pytket.Circuit(n_qubits)
    circuit = gen_term_sequence_circuit(operator, initial_circ)

    Transform.UCCSynthesis(method, CXConfigType.Tree).apply(circuit)
    tket_arch = Architecture([(e1, e2) for (e1, e2) in topo.to_nx.edges()])
    unit = CompilationUnit(circuit)
    passes = SequencePass(
        [
            PlacementPass(GraphPlacement(tket_arch)),
            RoutingPass(tket_arch),
        ]
    )
    passes.apply(unit)
    circ_out = unit.circuit
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
    operator = pp_to_operator(pp)
    circ_out = synth_tket(operator, topo, PauliSynthStrat.Sets)
    return get_ops_count(circ_out)


def synth_pp_tket_uccs_pair(
    pp: PauliPolynomial, topo: Topology, prefix="tket_uccs_pair"
):
    operator = pp_to_operator(pp)
    circ_out = synth_tket(operator, topo, PauliSynthStrat.Pairwise)
    return get_ops_count(circ_out)


def synth_pp_pauliopt_ucc(pp: PauliPolynomial, topo: Topology, prefix="pauliopt_ucc"):
    pp = simplify_pauli_polynomial(pp, allow_acs=True)
    synthesizer = PauliSynthesizer(pp, SynthMethod.UCCDS, topo)
    synthesizer.synthesize()
    return get_ops_count(synthesizer.circ_out_qiskit)


def synth_pp_pauliopt_steiner_nc(
    pp: PauliPolynomial, topo: Topology, prefix="pauliopt_steiner_nc"
):
    pp = simplify_pauli_polynomial(pp, allow_acs=True)
    synthesizer = PauliSynthesizer(pp, SynthMethod.STEINER_GRAY_NC, topo)
    synthesizer.synthesize()
    return get_ops_count(synthesizer.circ_out_qiskit)


def synth_pp_pauliopt_steiner_clifford(
    pp: PauliPolynomial, topo: Topology, prefix="pauliopt_steiner_clifford"
):
    pp = simplify_pauli_polynomial(pp, allow_acs=True)
    synthesizer = PauliSynthesizer(pp, SynthMethod.STEINER_GRAY_CLIFFORD, topo)
    synthesizer.synthesize()
    return get_ops_count(synthesizer.circ_out_qiskit)


def synth_pp_pauliopt_divide_conquer(
    pp: PauliPolynomial, topo: Topology, prefix="pauliopt_divide_conquer"
):
    synthesizer = PauliSynthesizer(pp, SynthMethod.DIVIDE_AND_CONQUER, topo)
    synthesizer.synthesize()
    return get_ops_count(synthesizer.circ_out_qiskit)


def paulihedral_rep_from_paulipolynomial(pp: PauliPolynomial):
    parr = []
    for p_gadget in pp:
        p_str = "".join([ps.value for ps in p_gadget.paulis])
        coeff = (
            p_gadget.angle if isinstance(p_gadget.angle, float) else np.random.rand()
        )

        p_string_ph = pauliString(p_str, coeff=coeff)
        parr.append([p_string_ph])
    return parr


def topology_to_ph_graph(topo: Topology, dist_comp=True):
    n = topo.num_qubits
    G = nx.adjacency_matrix(topo.to_nx).todense()
    C = topo._dist

    return pGraph(G, C)


def synth_pp_paulihedral(pp: PauliPolynomial, topo: Topology, prefix=None):
    graph = topology_to_ph_graph(topo)
    parr = paulihedral_rep_from_paulipolynomial(pp)

    lnq = len(parr[0][0])
    length = lnq // 2

    a1 = depth_oriented_scheduling(parr, length=length, maxiter=100)

    qc = synthesis_SC.block_opt_SC(a1, graph=graph, arch=None)
    return get_ops_count(qc)


def synth_pp_naive(pp: PauliPolynomial, topo: Topology, prefix="naive"):
    circ_out = pp.to_circuit(topo).to_qiskit()
    with open(prefix, "wb") as handle:
        pickle.dump(circ_out, handle)
    return get_ops_count(circ_out)


def get_ops_count(qc: QuantumCircuit):
    qc = transpile(qc, basis_gates=["u3", "cx"])
    count = {"cx": 0, "u3": 0, "depth": 0}
    ops = qc.count_ops()
    if "cx" in ops.keys():
        count["cx"] += ops["cx"]
    if "swap" in ops.keys():
        count["cx"] += ops["swap"] * 3
    count["u3"] = ops["u3"]
    count["depth"] = qc.depth()
    count["2q_depth"] = get_2q_depth(qc)
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

    return Topology.from_qiskit_backend(topo_kind)


SYNTHESIS_METHODS = {
    "tket_uccs_set": synth_pp_tket_uccs_set,
    "tket_uccs_pair": synth_pp_tket_uccs_pair,
    "paulihedral": synth_pp_paulihedral,
    "pauliopt_steiner_nc": synth_pp_pauliopt_steiner_nc,
    "pauliopt_ucc": synth_pp_pauliopt_ucc,
    "pauliopt_divide_conquer": synth_pp_pauliopt_divide_conquer,
    "pauliopt_steiner_clifford": synth_pp_pauliopt_steiner_clifford,
    "naive": synth_pp_naive,
}


def random_pauli_experiment(
    backend_name="vigo", nr_input_gates=100, nr_steps=5, df_name="data/random"
):
    backend, output_csv = get_backend_and_df_name(backend_name, df_name=df_name)
    if backend not in ["complete", "line"]:
        num_qubits = backend.configuration().num_qubits
    else:
        num_qubits = int(backend_name.split("_")[1])

    topo = get_topo_kind(backend, num_qubits)

    df = pd.DataFrame(
        columns=[
            "n_rep",
            "num_qubits",
            "n_gadgets",
            "method",
            "h",
            "s",
            "cx",
            "time",
            "depth",
            "2q_depth",
        ]
    )
    circuit_folder = "datasets/pauli_experiments/"
    naive_circuit_folder = "datasets/pauli_experiments/naive_circuits/"
    topo_folder = os.path.join(circuit_folder, backend_name)
    circuit_topo_folder = os.path.join(naive_circuit_folder, backend_name)
    os.makedirs(topo_folder, exist_ok=True)
    os.makedirs(circuit_topo_folder, exist_ok=True)
    for num_gadgets, i in itertools.product(
        range(1, nr_input_gates, nr_steps), range(20)
    ):
        circuit_file = os.path.join(
            topo_folder, f"pp_{backend_name}_{num_gadgets:03}_{i:02}.pickle"
        )
        naive_circuit = os.path.join(
            circuit_topo_folder, f"nc_{backend_name}_{num_gadgets:03}_{i:02}.pickle"
        )
        pp = create_random_pauli_polynomial(num_qubits, num_gadgets)
        pp = simplify_pauli_polynomial(pp, allow_acs=True)

        with open(circuit_file, "wb") as handle:
            pickle.dump(pp, handle)

        for synth, synth_method in SYNTHESIS_METHODS.items():
            # print(f"Synth: {synth_method}")
            column = {
                "n_rep": i,
                "num_qubits": num_qubits,
                "n_gadgets": num_gadgets,
                "method": synth,
            } | synth_method(pp, topo, prefix=naive_circuit)

            df.loc[len(df)] = column
            df.to_csv(output_csv)

    df.to_csv(output_csv)


def create_csv_header():
    header = [
        "n_rep",
        "num_qubits",
        "n_gadgets",
        "method",
        "cx",
        "depth",
        "2q_depth",
        "time",
    ]
    return header


def get_lock(new_lock):
    global lock
    lock = new_lock


def threaded_random_pauli_experiment(
    backend_name="vigo", nr_input_gates=100, nr_steps=5, df_name="data/random"
):
    manager = Manager()
    backend, output_csv = get_backend_and_df_name(
        backend_name, df_name=df_name)

    df = pd.DataFrame(
        columns=[
            "n_rep",
            "num_qubits",
            "n_gadgets",
            "method",
            "h",
            "s",
            "cx",
            "time",
            "depth",
            "2q_depth",
        ]
    )
    circuit_folder = "datasets/pauli_experiments/"
    naive_circuit_folder = "datasets/pauli_experiments/naive_circuits/"
    topo_folder = os.path.join(circuit_folder, backend_name)
    circuit_topo_folder = os.path.join(naive_circuit_folder, backend_name)
    os.makedirs(topo_folder, exist_ok=True)
    os.makedirs(circuit_topo_folder, exist_ok=True)

    gadget_iter = range(1, nr_input_gates, nr_steps)
    exp_iter = range(20)
    synth = SYNTHESIS_METHODS.items()

    if backend not in ["complete", "line"]:
        num_qubits = backend.configuration().num_qubits
    else:
        num_qubits = int(backend_name.split("_")[1])

    print("Synthesizing all PauliPolynomials")
    pp_dict = manager.dict()
    for num_gadgets, i in product(gadget_iter, exp_iter):
        pp = create_random_pauli_polynomial(num_qubits, num_gadgets)
        pp = simplify_pauli_polynomial(pp, allow_acs=True)
        pp_dict[(num_qubits, i, num_gadgets)] = pp

        circuit_file = os.path.join(
            topo_folder, f"pp_{backend_name}_{num_gadgets:03}_{i:02}.pickle"
        )
        naive_circuit = os.path.join(
            circuit_topo_folder, f"nc_{backend_name}_{num_gadgets:03}_{i:02}.pickle"
        )

        with open(circuit_file, "wb") as handle:
            pickle.dump(pp, handle)
    print(
        "Done synthesizing",
        len(pp_dict),
        "PauliPolynomials of size",
        get_size(pp_dict),
        "bytes",
    )

    arguments = product(gadget_iter, exp_iter, synth)
    arguments = zip(
        arguments,
        repeat((backend_name, df_name, topo_folder, circuit_topo_folder, pp_dict)),
    )

    # Create a Queue for the workers to work from
    q = manager.Queue()
    for args in arguments:
        q.put(args)
    original_size = q.qsize()
    n_workers = os.cpu_count() * 2
    print("Created task Queue on length", original_size)

    print("Creating csv fill at", datetime.now())
    df = pd.DataFrame({c: [] for c in create_csv_header()})
    with open(output_csv, "wb") as f:
        df.to_csv(f, header=create_csv_header(), index=False)
    print("Starting the pool with ", n_workers, "workers")
    lock = Lock()
    p = Pool(n_workers, initializer=get_lock, initargs=(lock,))
    for _ in range(n_workers):
        results = p.apply_async(pooled_function, [q])
    p.close()
    p.join()
    if not q.empty():
        print("Something went wrong, Queue is not empty", q.qsize())
    print("Done")


def pooled_function(q: Queue):
    while True:
        q.qsize() % 100 == 0 and print(".", end="", flush=True)
        try:
            q_data = q.get(timeout=10)
            if q_data is None:
                return
        except:  # Q is empty
            return
        exp_data, fixed_data = q_data
        num_gadgets, i, synth_data = exp_data
        synth_name, synth_method = synth_data
        backend_name, df_name, topo_folder, circuit_topo_folder, pp_dict = fixed_data

        backend, output_csv = get_backend_and_df_name(backend_name, df_name=df_name)
        if backend not in ["complete", "line"]:
            num_qubits = backend.configuration().num_qubits
        else:
            num_qubits = int(backend_name.split("_")[1])
        topo = get_topo_kind(backend, num_qubits)

        pp = pp_dict[(num_qubits, i, num_gadgets)]

        start = datetime.now()
        count_dict = synth_method(pp, topo)
        time_passed = (datetime.now() - start).total_seconds()

        column = {
            "n_rep": i,
            "num_qubits": num_qubits,
            "n_gadgets": num_gadgets,
            "time": time_passed,
            "method": synth_name,
        } | count_dict
        lock.acquire()
        df = pd.DataFrame([{c: column[c] for c in create_csv_header()}])
        with open(output_csv, "ab") as f_ptr:
            df.to_csv(f_ptr, header=False, index=False)
        lock.release()


if __name__ == "__main__":
    df_name = "data/pauli/random/random"
    print("Experiment: quito")
    threaded_random_pauli_experiment(
        backend_name="quito", nr_input_gates=200, nr_steps=20, df_name=df_name
    )
    print("Experiment: complete_5")
    threaded_random_pauli_experiment(
        backend_name="complete_5", nr_input_gates=200, nr_steps=20, df_name=df_name
    )

    print("Experiment: nairobi")
    threaded_random_pauli_experiment(
        backend_name="nairobi", nr_input_gates=300, nr_steps=20, df_name=df_name
    )
    print("Experiment: complete_7")
    threaded_random_pauli_experiment(
        backend_name="complete_7", nr_input_gates=300, nr_steps=20, df_name=df_name
    )

    print("Experiment: guadalupe")
    threaded_random_pauli_experiment(
        backend_name="guadalupe", nr_input_gates=400, nr_steps=20, df_name=df_name
    )
    print("Experiment: complete_16")
    threaded_random_pauli_experiment(
        backend_name="complete_16", nr_input_gates=400, nr_steps=20, df_name=df_name
    )

    print("Experiment: mumbai")
    threaded_random_pauli_experiment(
        backend_name="mumbai", nr_input_gates=800, nr_steps=40, df_name=df_name
    )
    print("Experiment: complete_27")
    threaded_random_pauli_experiment(
        backend_name="complete_27", nr_input_gates=800, nr_steps=40, df_name=df_name
    )

    print("Experiment: ithaca")
    threaded_random_pauli_experiment(
        backend_name="ithaca", nr_input_gates=2000, nr_steps=100, df_name=df_name
    )

    print("Experiment: complete_65")
    threaded_random_pauli_experiment(
        backend_name="complete_65", nr_input_gates=2000, nr_steps=100, df_name=df_name
    )

    print("Experiment: brisbane")
    threaded_random_pauli_experiment(
        backend_name="brisbane", nr_input_gates=10000, nr_steps=400, df_name=df_name
    )
    print("Experiment: complete_127")
    threaded_random_pauli_experiment(
        backend_name="complete_127", nr_input_gates=10000, nr_steps=400, df_name=df_name
    )
