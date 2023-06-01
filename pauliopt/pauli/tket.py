import numpy as np

from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.utils import X, Y, Z, I
from pauliopt.topologies import Topology


def synth_tket(pp: PauliPolynomial, topo: Topology = None,
               method="PauliSimp",
               kwargs=None):
    """
    Synthesize a PauliPolynomial using tket. This function requires tket to be installed.
    :param pp: PauliPolynomial to synthesize
    :param topo: Topology to synthesize on
    :param method: Method to use for synthesis:
        - PauliSimp: Simplify the PauliPolynomial using the PauliSimp pass and route the
                        resulting circuit afterwards
        - GuidedPauliSimp: Simplify the PauliPolynomial using the GuidedPauliSimp pass and
                            route the resulting circuit afterwards
        - PauliSquash: Simplify the PauliPolynomial using the PauliSquash pass and route
                        the resulting circuit afterwards
        - list of passes: Apply the passes in the list in order constructing your own
                            synthesis strategy with tket
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

    coupling_graph = [e for e in topo.to_nx.edges()]
    tket_arch = pytket.routing.Architecture(coupling_graph)

    unit = CompilationUnit(circuit, [ConnectivityPredicate(tket_arch)])

    if method == "PauliSimp":
        # get the strat from kwargs
        strat = kwargs.get("strat", PauliSynthStrat.Pairwise)
        passes = SequencePass([
            DecomposeBoxes(),
            PauliSimp(strat=strat),
            RoutingPass(tket_arch),
        ])
    elif method == "PauliSquash":
        strat = kwargs.get("strat", PauliSynthStrat.Pairwise)
        passes = SequencePass([
            DecomposeBoxes(),
            PauliSquash(strat=strat),
            RoutingPass(tket_arch),
        ])
    elif method == "GuidedPauliSimp":
        strat = kwargs.get("strat", PauliSynthStrat.Pairwise)
        passes = SequencePass([
            DecomposeBoxes(),
            GuidedPauliSimp(strat=strat),
            RoutingPass(tket_arch),
        ])
    elif isinstance(method, list):
        passes = SequencePass(method)
    else:
        raise Exception(f"Unsupported Method: {method}")

    passes.apply(unit)
    tk_to_qiskit(unit.circui)
