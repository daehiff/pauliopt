from pauliopt.circuits import Circuit
from bidict import bidict


class CNOTComb:
    pass


def circuit_to_cnot_comb(circuit: Circuit) -> CNOTComb:
    """
    Convert a circuit to a comb
    :param circuit:
    :param topology:
    :return:
    """

    n_qubits = circuit.n_qubits

    open_holes = {}
    hole_plugs = {}
    hole_qubit_mappings = bidict()
    moving_qubit_mappings = bidict()

    new_to_old_qubit_mappings = {}

    CNOTs_for_comb = []

    for gate in circuit._gates:
        if gate.name != "CX":
            # check if there is a mapping for the gate's qubit
            qubit = gate.qubits[0]
            if qubit in moving_qubit_mappings.keys():
                qubit = moving_qubit_mappings[qubit]

            # insert the gate into the open holes
            if qubit not in open_holes.keys():
                open_holes[qubit] = [gate]
            else:
                open_holes[qubit].append(gate)
        else:
            # Convert old qubit to new qubit using moving mappings
            if gate.target in moving_qubit_mappings.keys():
                gate.target = moving_qubit_mappings[gate.target]
            if gate.control in moving_qubit_mappings.keys():
                gate.control = moving_qubit_mappings[gate.control]

            if gate.target in open_holes.keys() or gate.control in open_holes.keys():
                for qubit in [gate.target, gate.control]:
                    if qubit in open_holes.keys():
                        # Creating a new qubit and link it to this one
                        hole_qubit_mappings[qubit] = n_qubits

                        # Check to see if we are mapping a qubit that has already been mapped
                        # Allowing us to have a mapping from the initial qubit to the current one
                        if qubit in moving_qubit_mappings.inverse.keys():
                            temp_dict = bidict(
                                {moving_qubit_mappings.inverse.pop(qubit): n_qubits}
                            )
                            moving_qubit_mappings.update(temp_dict)
                            new_to_old_qubit_mappings.update(temp_dict.inverse)
                        # If this mapping isn't in the moving mapping added
                        if (
                            qubit not in moving_qubit_mappings.keys()
                            and n_qubits not in moving_qubit_mappings.inverse.keys()
                        ):
                            temp_dict = bidict({qubit: n_qubits})
                            moving_qubit_mappings.update(temp_dict)
                            new_to_old_qubit_mappings.update(temp_dict.inverse)
                        hole_plugs[hole_qubit_mappings[qubit]] = open_holes.pop(qubit)

                        n_qubits = n_qubits + 1

                        # Need to do this again to correct for newly created mappings
                        # Convert old qubit to new qubit using moving mappings
                        if gate.target in hole_qubit_mappings.keys():
                            gate.target = hole_qubit_mappings[gate.target]
                        if gate.control in hole_qubit_mappings.keys():
                            gate.control = hole_qubit_mappings[gate.control]

            CNOTs_for_comb.append(gate)

    for qubit in list(open_holes.keys()):
        if qubit not in hole_qubit_mappings.keys():
            hole_qubit_mappings[qubit] = n_qubits
            # Check to see if we are mapping a qubit that has already been mapped
            # Allowing us to have a mapping from the initial qubit to the current one
            if qubit in moving_qubit_mappings.inverse.keys():
                temp_dict = bidict({moving_qubit_mappings.inverse.pop(qubit): n_qubits})
                moving_qubit_mappings.update(temp_dict)
                new_to_old_qubit_mappings.update(temp_dict.inverse)
            # If this mapping isn't in the moving mapping added
            if (
                qubit not in moving_qubit_mappings.keys()
                and n_qubits not in moving_qubit_mappings.inverse.keys()
            ):
                temp_dict = bidict({qubit: n_qubits})
                moving_qubit_mappings.update(temp_dict)
                new_to_old_qubit_mappings.update(temp_dict.inverse)
            hole_plugs[hole_qubit_mappings[qubit]] = open_holes.pop(qubit)
            n_qubits = n_qubits + 1

    print(CNOTs_for_comb)
