OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
s q[6];
h q[1];
h q[3];
h q[0];
h q[6];
h q[0];
h q[6];
h q[0];
cx q[2],q[3];
h q[2];
cx q[0],q[3];
h q[1];
cx q[1],q[0];
cx q[0],q[4];
s q[0];
cx q[6],q[5];
h q[2];
s q[2];
cx q[0],q[3];
h q[6];
s q[2];
