OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
cx q[0],q[1];
s q[2];
h q[4];
cx q[1],q[0];
cx q[1],q[3];
cx q[0],q[2];
h q[1];
s q[1];
s q[1];
cx q[0],q[3];
s q[4];
s q[4];
cx q[4],q[3];
h q[4];
h q[0];
s q[4];
h q[2];
h q[2];
s q[0];
cx q[4],q[3];
h q[4];
