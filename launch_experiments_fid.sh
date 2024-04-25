#!/bin/bash

sbatch -p All --export=ALL,ALG="default",QUBITS="6",GADGETS="160" -o "fid_default_6_160.out" run_experiment.sh
#sbatch -p All --export=ALL,ALG="default",QUBITS="10",GADGETS="630" -o "fid_default_10_630.out" run_experiment.sh

sbatch -p All --export=ALL,ALG="UCCSD",QUBITS="6",GADGETS="160" -o "fid_UCCSD_6_160.out" run_experiment.sh
#sbatch -p All --export=ALL,ALG="UCCSD",QUBITS="10",GADGETS="630" -o "fid_UCCSD_10_630.out" run_experiment.sh

sbatch -p All --export=ALL,ALG="paulihedral",QUBITS="6",GADGETS="160" -o "fid_paulihedral_6_160.out" run_experiment.sh
#sbatch -p All --export=ALL,ALG="paulihedral",QUBITS="10",GADGETS="630" -o "fid_paulihedral_10_630.out" run_experiment.sh

sbatch -p All --export=ALL,ALG="PSGS",QUBITS="6",GADGETS="160" -o "fid_PSGS_6_160.out" run_experiment.sh
#sbatch -p All --export=ALL,ALG="PSGS",QUBITS="10",GADGETS="630" -o "fid_PSGS_10_630.out" run_experiment.sh

