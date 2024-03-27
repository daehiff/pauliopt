from enum import Enum

from pauliopt.pauli.pauli_circuit import PauliCircuit
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.pauli.synth.steiner_gray_synth import pauli_polynomial_steiner_gray_clifford
from pauliopt.pauli.utils import apply_permutation, verify_equality
from pauliopt.topologies import Topology

from pauliopt.pauli.synth import anneal, divide_and_conquer, \
    pauli_polynomial_steiner_gray_nc, uccds


class SynthMethod(Enum):
    ANNEAL = "anneal"
    DIVIDE_AND_CONQUER = "divide_and_conquer"
    STEINER_GRAY_NC = "steiner_gray_nc"
    STEINER_GRAY_CLIFFORD = "steiner_gray_clifford"
    UCCDS = "uccds"


class PauliSynthesizer:
    def __init__(self, pp: PauliPolynomial, method: SynthMethod, topology: Topology):
        self.topology: Topology = topology
        self.pp: PauliPolynomial = pp
        self.method: SynthMethod = method
        self.qubit_placement: list = None
        self.gadget_placement: list = None
        self.circ_out_: PauliCircuit = None

    @property
    def circ_out_qiskit(self):
        return self.circ_out_.to_qiskit()

    def synthesize(self):
        method = self.method
        pp = self.pp.copy()
        if method == SynthMethod.ANNEAL:
            circ_out, gadget_perm, perm = \
                anneal(pp, self.topology)
        elif method == SynthMethod.DIVIDE_AND_CONQUER:
            circ_out, gadget_perm, perm = \
                divide_and_conquer(pp, self.topology)
        elif method == SynthMethod.STEINER_GRAY_NC:
            circ_out, gadget_perm, perm = \
                pauli_polynomial_steiner_gray_nc(pp, self.topology)
            circ_out.apply_permutation(perm)
        elif method == SynthMethod.STEINER_GRAY_CLIFFORD:
            circ_out, gadget_perm, perm = \
                pauli_polynomial_steiner_gray_clifford(pp, self.topology)
            circ_out.apply_permutation(perm)
        elif method == SynthMethod.UCCDS:
            circ_out, gadget_perm, perm = \
                uccds(pp, self.topology)
            circ_out.apply_permutation(perm)
        else:
            raise ValueError("Unknown method: {}".format(method))

        self.qubit_placement = perm
        self.gadget_placement = gadget_perm
        self.circ_out_ = circ_out

    def check_connectivity_predicate(self):
        G = self.topology.to_nx

        if self.circ_out_qiskit is None:
            raise ValueError("No circuit has been synthesized yet")
        register = self.circ_out_qiskit.qubits
        for gate in self.circ_out_qiskit:
            if gate.operation.num_qubits == 2:
                ctrl, target = gate.qubits
                ctrl, target = register.index(ctrl), register.index(target)
                if not G.has_edge(ctrl, target):
                    return False
        return True

    def check_circuit_equivalence(self):
        pp_ = PauliPolynomial(self.pp.num_qubits)
        pp_.pauli_gadgets = [self.pp[i].copy() for i in self.gadget_placement]
        circ = self.circ_out_qiskit.copy()
        circ = apply_permutation(circ, self.qubit_placement)

        pp_circ = pp_.to_qiskit()
        return verify_equality(circ, pp_circ)

    def get_operator(self):
        import qiskit
        from pytket.extensions.qiskit.backends import aer

        unitarysimulator = aer.Aer.get_backend("unitary_simulator")
        circ = self.circ_out_qiskit.copy()
        circ = apply_permutation(circ, self.qubit_placement)

        result = qiskit.execute(circ, unitarysimulator).result()
        U = result.get_unitary()
        return U


def synthesize(pp: PauliPolynomial, topo: Topology, method: SynthMethod):
    synthesizer = PauliSynthesizer(pp, method, topo)
    synthesizer.synthesize()
    return synthesizer.circ_out_qiskit, synthesizer.qubit_placement
