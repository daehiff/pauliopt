// i 0 1 2
// o 0 1 2
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[2];
cx q[0], q[1];
sdg q[0];
cx q[2], q[1];
s q[1];
cx q[2], q[0];
z q[0];
h q[1];
h q[2];
sdg q[2];
cx q[0], q[1];
s q[0];
h q[1];
s q[1];
cx q[2], q[0];
sdg q[0];
x q[1];
h q[2];
