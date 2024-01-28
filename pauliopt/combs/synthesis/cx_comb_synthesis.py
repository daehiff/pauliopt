import galois
import networkx as nx
import numpy as np

from pauliopt.circuits import Circuit
from pauliopt.cnots.synthesis.perm_row_col import row_col_iteration
from pauliopt.combs.comb_conversion import circuit_to_cnot_comb
from pauliopt.gates import CX
from pauliopt.topologies import Topology


def is_cutting(vertex, g):
    # TODO this is duplicate from clifford synthesis, copy refactor it onto topology
    return vertex in nx.articulation_points(g)


def get_non_cutting_vertex(topo: Topology, rows_to_eliminate):
    G = topo.to_nx
    G = G.subgraph(rows_to_eliminate)

    non_cutting_vertices = []
    for node in G.nodes:
        if not is_cutting(node, G):
            non_cutting_vertices.append(node)
    return non_cutting_vertices


def extract_sub_matrix(matrix, index_list):
    new_matrix = galois.GF2(
        np.zeros((len(index_list), matrix.shape[0]), dtype=np.uint8)
    )
    for row in range(new_matrix.shape[0]):
        for col in range(new_matrix.shape[1]):
            new_matrix[row][col] = matrix[index_list[row]][col]
    return new_matrix


def insert_sub_matrix(matrix, sub_matrix, index_list):
    for row in range(sub_matrix.shape[0]):
        for col in range(sub_matrix.shape[1]):
            matrix[index_list[row]][col] = sub_matrix[row][col]
    return matrix


def next_elimination(
    qubit_dependence, qubits_in_matrix, topo: Topology, rows_to_eliminate
):
    # Use the qubit_dependence
    possible_eliminations = set(qubit_dependence.keys())
    # It's easier to find all the qubits that can't be eliminated by seeing all the qubits
    # that depend on currently unavailable qubits

    # I need all the elements in possible_eliminations that aren't in qubits_in_matrix
    unavailable_qubits = (
        possible_eliminations ^ set(qubits_in_matrix)
    ) & possible_eliminations
    # Generate a set of all the qubits that depend on currently unavailable qubits
    impossible_eliminations = unavailable_qubits.copy()
    for q in unavailable_qubits:
        impossible_eliminations = impossible_eliminations.union(qubit_dependence[q])
    # possible_eliminations now becomes all the qubits not in impossible eliminations
    possible_eliminations = possible_eliminations ^ impossible_eliminations
    # Find non-cutting vertices
    non_cutting_qubits = get_non_cutting_vertex(topo, rows_to_eliminate)

    eliminate = np.random.choice(
        [
            elim
            for elim in possible_eliminations
            if (elim >= len(qubits_in_matrix) or elim in non_cutting_qubits)
        ]
    )
    return eliminate


def comb_row_col(circuit, arch):
    circuit_out = Circuit(circuit.n_qubits)

    comb = circuit_to_cnot_comb(circuit)

    g_matrix = comb.matrix.copy()

    qubit_dependence = comb.qubit_dependence

    # 2 identify a set of extractiable qubits (qubits that can be eliminated)
    qubits_in_matrix = []
    old_to_new_qubits = dict([(i, []) for i in range(circuit.n_qubits)])
    for virtual_qubit in comb.new_to_old_qubit_mappings.keys():
        old_to_new_qubits[comb.new_to_old_qubit_mappings[virtual_qubit]].append(
            virtual_qubit
        )
    for logical_qubit in old_to_new_qubits.keys():
        if len(old_to_new_qubits[logical_qubit]) == 0:
            qubits_in_matrix.append(logical_qubit)
        else:
            qubits_in_matrix.append(max(old_to_new_qubits[logical_qubit]))

    rows_to_eliminate = list(range(circuit.n_qubits))
    # Qubits on the comb that haven't eliminated yet
    cols_to_eliminate = list(range(comb.n_qubits))
    while len(cols_to_eliminate) > 0:
        print(g_matrix)
        print(qubits_in_matrix)
        sub_matrix = extract_sub_matrix(g_matrix, qubits_in_matrix)
        print(sub_matrix)

        col_to_eliminate = next_elimination(
            qubit_dependence, qubits_in_matrix, arch, rows_to_eliminate
        )

        if col_to_eliminate in comb.new_to_old_qubit_mappings:
            row_to_eliminate = comb.new_to_old_qubit_mappings[col_to_eliminate]
        else:
            row_to_eliminate = col_to_eliminate

        sub_circuit = Circuit(circuit.n_qubits)

        def add_cnot(ctrl, trgt, matrix):
            matrix[ctrl, :] += matrix[trgt, :]
            sub_circuit.add_gate(CX(trgt, ctrl))

        row_col_iteration(
            sub_matrix,
            arch,
            row_to_eliminate,
            col_to_eliminate,
            rows_to_eliminate,
            cols_to_eliminate,
            add_cnot,
        )

        circuit_out._gates = sub_circuit._gates[::-1] + circuit_out._gates

        insert_sub_matrix(g_matrix, sub_matrix, qubits_in_matrix)

        # If the qubit just removed maps to another qubit via a hole
        # replace that qubit with the new qubit in the qubits_in_matrix list
        qubit_found = False
        qubit_loc = 0
        while not qubit_found:
            qubit = qubits_in_matrix[qubit_loc]
            if qubit == col_to_eliminate:
                qubit_found = True
                cols_to_eliminate.remove(col_to_eliminate)
                if qubit in comb.hole_plugs.keys():
                    circuit_out._gates = comb.hole_plugs.pop(qubit) + circuit_out._gates

                if qubit in comb.holes.inverse.keys():
                    qubits_in_matrix[qubit_loc] = comb.holes.inverse.pop(qubit)
                else:
                    if qubit in comb.new_to_old_qubit_mappings:
                        rows_to_eliminate.remove(comb.new_to_old_qubit_mappings[qubit])
                    else:
                        rows_to_eliminate.remove(qubit)

            qubit_loc += 1

        # Remove the qubit that has just been eliminated from the dependencies
        qubit_dependence.pop(col_to_eliminate)
        for qubit in qubit_dependence:
            if col_to_eliminate in qubit_dependence[qubit]:
                qubit_dependence[qubit].remove(col_to_eliminate)

    return circuit_out
