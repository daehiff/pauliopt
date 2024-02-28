// i 0 1 2 3
// o 0 1 2 3
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
s q[3];
cx q[0], q[3];
cx q[1], q[3];
x q[1];
cx q[2], q[3];
sdg q[3];
cx q[2], q[1];
h q[1];
cx q[2], q[3];
h q[3];
cx q[3], q[2];
h q[2];
cx q[1], q[0];
cx q[0], q[3];
y q[0];
