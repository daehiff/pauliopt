// i 0 1 2 3
// o 0 1 2 3
OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
s q[0];
z q[1];
cx q[2], q[3];
h q[0];
z q[2];
sdg q[3];
cx q[2], q[0];
h q[2];
cx q[0], q[1];
s q[0];
s q[2];
cx q[1], q[3];
sdg q[1];
cx q[3], q[2];
