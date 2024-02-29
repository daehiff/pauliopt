// i 0 1 2
// o 0 1 2
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
z q[1];
cx q[0], q[2];
h q[0];
cx q[1], q[2];
s q[0];
sdg q[1];
z q[2];
h q[1];
