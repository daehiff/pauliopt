// i 0 1 2
// o 0 1 2
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
s q[2];
cx q[0], q[2];
cx q[2], q[1];
s q[1];
cx q[1], q[2];
y q[2];
cx q[2], q[1];
h q[2];
