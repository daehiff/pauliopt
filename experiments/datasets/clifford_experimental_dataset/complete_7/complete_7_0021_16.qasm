OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
h q[1];
cx q[0],q[6];
cx q[2],q[0];
h q[0];
s q[0];
cx q[6],q[0];
h q[6];
s q[3];
h q[4];
cx q[4],q[6];
cx q[4],q[5];
h q[4];
s q[1];
cx q[0],q[5];
s q[0];
s q[0];
cx q[4],q[5];
s q[0];
h q[1];
cx q[6],q[1];
s q[1];
