OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
cx q[6],q[5];
cx q[0],q[2];
cx q[5],q[4];
s q[1];
s q[5];
h q[3];
s q[5];
cx q[5],q[2];
h q[0];
h q[3];
h q[6];
s q[4];
s q[6];
cx q[6],q[3];
h q[1];
s q[5];
cx q[0],q[3];
s q[6];
s q[4];
s q[4];
s q[6];
