// i 0 1 2
// o 0 1 2
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
sdg q[0];
s q[1];
s q[2];
h q[0];
cx q[1], q[2];
x q[0];
h q[1];
h q[2];
sdg q[2];
cx q[1], q[0];
s q[0];
cx q[1], q[2];
