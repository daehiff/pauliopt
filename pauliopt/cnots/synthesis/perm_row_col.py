from typing import Tuple, List

import networkx as nx
import numpy as np

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


def get_rows_to_eliminate(p_matrix, qubits_to_process, columns_to_eliminate, row, col):
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
        qubits_to_process (List[int]): The qubits that are not yet eliminated
        columns_to_eliminate (List[int]): The columns that are not yet eliminated
        row (int): The row to eliminate
        col (int): The column to eliminate

    Returns:
        List[int]: The rows to eliminate
    """

    try:
        import galois
    except ImportError:
        raise ImportError(
            "The perm-row-col synthesis algorithm requires the galois library."
        )
    qubits_to_process_ = [i for i in qubits_to_process if i != row]
    columns_to_eliminate_ = [i for i in columns_to_eliminate if i != col]
    submatrix = [
        [p_matrix[i, j] for i in qubits_to_process if i != row]
        for j in columns_to_eliminate
        if j != col
    ]
    A = galois.GF(2)(submatrix)
    A_inv = np.linalg.inv(A)
    B = galois.GF(2)(
        [p_matrix[row, i] for i in columns_to_eliminate if i != col], dtype=np.int32
    )
    # A = p_matrix[columns_to_eliminate_, :][:, qubits_to_process_].T
    # A_inv = np.linalg.inv(A)
    # B = p_matrix[row, columns_to_eliminate_]
    X1 = np.matmul(A_inv, B)
    # Add the row that we removed back in for easier indexing.
    X = np.insert(X1, qubits_to_process.index(row), 1)

    # subselect the columns that we need to process (that are one)
    return [qubits_to_process[i] for i in range(len(qubits_to_process)) if X[i] == 1]


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

    def add_cnot(ctrl, trgt):
        parity_map.append_cnot(trgt, ctrl)
        cnot_circuit.add_gate(CX(ctrl, trgt))

    qubits_to_process = list(range(topology.num_qubits))
    columns_to_eliminate = list(range(topology.num_qubits))
    new_mapping = [-1] * topology.num_qubits
    while len(qubits_to_process) > 1:
        # Pick the pivot location
        possible_qubits = topology.non_cutting_qubits(qubits_to_process)
        row = choose_row(possible_qubits, parity_map)
        col = choose_column(
            columns_to_eliminate,
            row,
            parity_map,
            topology.num_qubits,
            reallocate=reallocate,
        )

        # Reduce the column
        col_nodes = [i for i in qubits_to_process if parity_map[i, col] == 1]
        if parity_map[row, col] == 0:
            col_nodes.append(row)

        steiner_tree = topology.steiner_tree(col_nodes, qubits_to_process)
        if len(col_nodes) > 1:
            traversal = list(reversed(list(nx.bfs_edges(steiner_tree, source=row))))
            for parent, child in traversal:
                if parity_map[parent, col] == 0:
                    add_cnot(parent, child)
                    # parity_map.append_cnot(child, parent)

            for parent, child in traversal:
                add_cnot(child, parent)

        assert np.sum(np.asarray(parity_map[:, col])) == 1
        assert parity_map[row, col] == 1

        # Reduce the row
        ones_in_the_row = [i for i in columns_to_eliminate if parity_map[row, i] == 1]
        if len(ones_in_the_row) > 1:
            row_nodes = get_rows_to_eliminate(
                parity_map.matrix, qubits_to_process, columns_to_eliminate, row, col
            )
            steiner_tree = topology.steiner_tree(row_nodes, qubits_to_process)
            traversal = list(nx.bfs_edges(steiner_tree, source=row))
            for parent, child in traversal:
                if child not in row_nodes:
                    add_cnot(parent, child)
            for parent, child in reversed(traversal):
                add_cnot(parent, child)

        assert np.sum(np.asarray(parity_map[row, :])) == 1
        assert parity_map[row, col] == 1

        qubits_to_process.remove(row)
        columns_to_eliminate.remove(col)
        new_mapping[col] = row
    new_mapping[columns_to_eliminate[0]] = qubits_to_process[0]

    cnot_circuit.final_mapping = new_mapping
    return cnot_circuit
