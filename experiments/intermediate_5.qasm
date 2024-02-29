// i 0 1 2
// o 0 1 2
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
sdg q[0];
h q[1];
h q[2];
s q[1];
cx q[2], q[0];
h q[2];
cx q[0], q[1];
sdg q[1];
cx q[2], q[0];
sdg q[0];
h q[1];
sdg q[2];
y q[1];
h q[2];
