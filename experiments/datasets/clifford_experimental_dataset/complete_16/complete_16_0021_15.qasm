OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
s q[9];
h q[14];
h q[6];
h q[2];
h q[11];
s q[8];
cx q[11],q[0];
h q[7];
h q[5];
h q[5];
s q[11];
h q[15];
h q[12];
cx q[4],q[6];
h q[10];
h q[3];
cx q[2],q[4];
h q[1];
cx q[8],q[0];
s q[14];
cx q[0],q[5];
