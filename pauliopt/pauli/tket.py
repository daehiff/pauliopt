import numpy as np

from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.utils import X, Y, Z, I
from pauliopt.topologies import Topology


def synth_tket(pp: PauliPolynomial, topo: Topology = None,
               method="PauliSimp",
               return_circuit=True,
               **kwargs):
    """
    Synthesize a PauliPolynomial using tket. This function requires tket to be installed.
    :param pp: PauliPolynomial to synthesize
    :param topo: Topology to synthesize on
    :param method: Method to use for synthesis:
        - PauliSimp: Simplify the PauliPolynomial using the PauliSimp pass and route the
                     resulting circuit afterwards
        - list of passes: Apply the passes in the list in order constructing your own
                          synthesis strategy with tket
    :param return_circuit: Wether to return a Qiskit circuit (sanatized, by swapping the
                           permutated qubits due to the routing process of tket) or the
                           compilation unit with circuit and placement

    :return: A pytket.Circuit or the result of applying the passes in the list if
    the circuit is not supposed to be swapped afterwards on the classical side

    Example:
    >>> from pauliopt.pauli.pauli_polynomial import PauliPolynomial
    >>> from pauliopt.pauli.utils import X, Y, Z, I
    >>> from pauliopt.utils import pi
    >>> from pauliopt.topologies import Topology
    >>> from pauliopt.pauli.tket import synth_tket
    >>> from pauliopt.pauli.pauli_gadget import PPhase
    >>> pp = PauliPolynomial(5)
    >>> pp >>= PPhase(pi/4) @ [X, Y, Z, I, X]
    >>> pp >>= PPhase(pi/2) @ [X, Y, Z, I, X]
    >>> pp >>= PPhase(pi/8) @ [X, X, Z, Z, X]
    >>> topo = Topology.line(5)
    >>> circuit = synth_tket(pp, topo, method="PauliSimp")
    >>> unit = synth_tket(pp, topo, return_circuit=False)
    """
    try:
        import pytket
        from pytket._tket.pauli import Pauli
        from pytket._tket.circuit import PauliExpBox
        from pytket._tket.transform import Transform
        from pytket.extensions.qiskit import tk_to_qiskit
        from pytket._tket.passes import SequencePass, PauliSimp, RoutingPass, \
            PauliSquash, GuidedPauliSimp, DecomposeBoxes
        from pytket._tket.predicates import CompilationUnit, ConnectivityPredicate
        from pytket.transform import PauliSynthStrat
        from pytket._tket.architecture import Architecture
        from pytket._tket.passes import PlacementPass
        from pytket._tket.placement import GraphPlacement

    except:
        raise Exception("In order for this function to work, please install:"
                        "pip install pytket pytket-qiskit")

    pauli_to_tket = {
        X: Pauli.X,
        Y: Pauli.Y,
        Z: Pauli.Z,
        I: Pauli.I
    }

    circuit = pytket.Circuit(pp.num_qubits)
    for gadget in pp.pauli_gadgets:
        circuit.add_pauliexpbox(
            PauliExpBox([pauli_to_tket[p] for p in gadget.paulis],
                        gadget.angle.to_qiskit / np.pi),
            list(range(pp.num_qubits)))

    tket_arch = Architecture([e for e in topo.to_nx.edges()])

    unit = CompilationUnit(circuit, [ConnectivityPredicate(tket_arch)])

    if method == "PauliSimp":
        # get the strat from kwargs
        strat = kwargs.get("strat", PauliSynthStrat.Pairwise)
        passes = SequencePass([
            PauliSimp(strat=strat),
            PlacementPass(GraphPlacement(tket_arch)),
            RoutingPass(tket_arch),
        ])
    elif isinstance(method, list):
        passes = SequencePass(method)
    else:
        raise Exception(f"Unsupported Method: {method}")

    passes.apply(unit)
    circ_out = unit.circuit
    if return_circuit:
        inv_map = {v: k for k, v in unit.final_map.items()}
        circ_out_ = pytket.Circuit(circ_out.n_qubits)
        for cmd in circ_out:
            if isinstance(cmd, pytket.circuit.Command):
                remaped_qubits = list(map(lambda node: inv_map[node], cmd.qubits))
                circ_out_.add_gate(cmd.op, remaped_qubits)
        circ_out = circ_out_

        return tk_to_qiskit(circ_out)
    else:
        return unit
