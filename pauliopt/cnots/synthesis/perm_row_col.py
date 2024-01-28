from typing import List

import networkx as nx
import numpy as np
from pyzx import Mat2

from pauliopt.circuits import Circuit
from pauliopt.cnots.parity_map import ParityMap
from pauliopt.gates import CX
from pauliopt.topologies import Topology


def choose_row(options, parity_map):
    return options[np.argmin([np.sum(np.asarray(parity_map[o])) for o in options])]


def choose_column(options, row, parity_map, n_qubits, reallocate=False):
    if reallocate:
        option_score = [
            np.sum(np.asarray(parity_map[:, o], dtype=np.int32))
            if parity_map[row, o] == 1
            else n_qubits
            for o in options
        ]
        return options[np.argmin(option_score)]
    return row


def get_rows_to_eliminate(
    p_matrix, rows_to_eliminate, cols_to_eliminate, choice_row, choice_col
):
    """
    Get the rows to elminiate the row of the parity map by computing a system of linear
    equations:
        A X = B
    where A is the matrix which contains all open rows and columns except
    from row and col,
    B is the row already eliminated and X will indicate which rows to
    eliminate in the later process

    Args:
        p_matrix (np.ndarray): The parity map matrix
        rows_to_eliminate (List[int]): The qubits that are not yet eliminated
        cols_to_eliminate (List[int]): The columns that are not yet eliminated
        choice_row (int): The row to eliminate
        choice_col (int): The column to eliminate

    Returns:
        List[int]: The rows to eliminate
    """

    try:
        import galois
    except ImportError:
        raise ImportError(
            "The perm-row-col synthesis algorithm requires the galois library."
        )
    A_ = Mat2(
        np.array(
            [
                [
                    int(p_matrix[row][col])
                    for row in rows_to_eliminate
                    if row != choice_row
                ]
                for col in cols_to_eliminate
                if col != choice_col
            ]
        )
    )
    B_ = Mat2(
        np.array(
            [
                [int(p_matrix[choice_row][col])]
                for col in cols_to_eliminate
                if col != choice_col
            ]
        )
    )
    A_.gauss(full_reduce=True, x=B_)
    X = A_.data.transpose().dot(B_.data).flatten()
    find_index = lambda i: [j for j in rows_to_eliminate if j != choice_row].index(i)
    return [
        i for i in rows_to_eliminate if i == choice_row or X[find_index(i)]
    ]  # This is S'


def row_col_iteration(
    matrix,
    topology: Topology,
    row: int,
    col: int,
    qubits_to_process: list,
    columns_to_eliminate: list,
    add_cnot,
):
    col_nodes = [i for i in qubits_to_process if matrix[i, col] == 1]
    if matrix[row, col] == 0:
        col_nodes.append(row)

    # Reduce the columns
    if len(col_nodes) > 1:
        steiner_tree = topology.steiner_tree(col_nodes, qubits_to_process)
        traversal = list(reversed(list(nx.bfs_edges(steiner_tree, source=row))))
        for parent, child in traversal:
            if matrix[parent, col] == 0:
                add_cnot(parent, child, matrix)

        for parent, child in traversal:
            add_cnot(child, parent, matrix)
    # assert np.sum(np.asarray(matrix[:, col])) == 1
    # assert matrix[row, col] == 1

    # Reduce the row
    ones_in_the_row = [i for i in columns_to_eliminate if matrix[row, i] == 1]
    if len(ones_in_the_row) > 1:
        row_nodes = get_rows_to_eliminate(
            matrix, qubits_to_process, columns_to_eliminate, row, col
        )
        steiner_tree = topology.steiner_tree(row_nodes, qubits_to_process)
        traversal = list(nx.bfs_edges(steiner_tree, source=row))
        for parent, child in traversal:
            if child not in row_nodes:
                add_cnot(parent, child, matrix)
        for parent, child in reversed(traversal):
            add_cnot(parent, child, matrix)

    # assert np.sum(np.asarray(matrix[row, :])) == 1
    # assert matrix[row, col] == 1


def _perm_row_col(
    parity_map: ParityMap,
    topology: Topology,
    parities_as_columns: bool = False,
    reallocate: bool = False,
) -> Circuit:
    """
    Synthesis of a parity map using the perm-row-col algorithm of [1].
    One can allow to reallocate qubits, i.e. the output permutation is not
    necessarily [0...n], by setting reallocate=True.

    Args:
        parity_map (ParityMap): The binary parity matrix
        topology (Topology): The target device topology
        parities_as_columns (bool): Whether the parities in the matrix are row-wise or
        column-wise. Defaults to False, i.e. row-wise.
        reallocate (bool, optional): Whether to qubits can re reallocated.

    Returns:
        Circuit:     The circuit that implements the parity map. The final mapping
                     is stored in the final_mapping attribute.


    -----
    [1] https://arxiv.org/abs/2205.00724
    """

    cnot_circuit = Circuit(parity_map.n_qubits)
    parity_map = parity_map.copy()

    if not parities_as_columns:
        # This synthesis technique only works when parities are columns.
        parity_map = parity_map.transpose()

    def add_cnot(ctrl, trgt, matrix):
        matrix[ctrl, :] += matrix[trgt, :]
        cnot_circuit.add_gate(CX(ctrl, trgt))

    qubits_to_process = list(range(topology.num_qubits))
    columns_to_eliminate = list(range(topology.num_qubits))
    new_mapping = [-1] * topology.num_qubits
    matrix = parity_map.matrix
    while len(qubits_to_process) > 1:
        # Pick the pivot location
        possible_qubits = topology.non_cutting_qubits(qubits_to_process)
        row = choose_row(possible_qubits, matrix)
        col = choose_column(
            columns_to_eliminate,
            row,
            matrix,
            topology.num_qubits,
            reallocate=reallocate,
        )

        row_col_iteration(
            matrix,
            topology,
            row,
            col,
            qubits_to_process,
            columns_to_eliminate,
            add_cnot,
        )

        qubits_to_process.remove(row)
        columns_to_eliminate.remove(col)
        new_mapping[col] = row
    new_mapping[columns_to_eliminate[0]] = qubits_to_process[0]

    cnot_circuit.final_mapping = new_mapping
    return cnot_circuit
