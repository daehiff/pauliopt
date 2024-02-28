// i 0 1 2 3
// o 0 1 2 3
OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
sdg q[0];
sdg q[2];
cx q[3], q[1];
sdg q[1];
h q[2];
cx q[0], q[3];
cx q[1], q[3];
cx q[2], q[0];
sdg q[0];
h q[1];
cx q[3], q[2];
cx q[2], q[0];
cx q[3], q[1];
h q[0];
cx q[3], q[2];
cx q[0], q[3];
cx q[1], q[2];
sdg q[2];
sdg q[3];
cx q[0], q[1];
