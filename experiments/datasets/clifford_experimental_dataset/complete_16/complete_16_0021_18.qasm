OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
h q[4];
s q[8];
s q[0];
cx q[12],q[13];
s q[0];
h q[7];
s q[11];
cx q[7],q[6];
s q[9];
s q[5];
s q[2];
cx q[1],q[12];
h q[5];
h q[13];
s q[10];
h q[0];
cx q[14],q[6];
cx q[4],q[5];
s q[1];
cx q[2],q[7];
s q[15];