// i 0 1 2
// o 0 1 2
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[0];
y q[1];
s q[2];
sdg q[0];
y q[2];
cx q[1], q[0];
h q[1];
