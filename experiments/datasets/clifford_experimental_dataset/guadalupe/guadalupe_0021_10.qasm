OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
s q[6];
cx q[8],q[12];
h q[0];
s q[9];
h q[14];
s q[15];
s q[5];
h q[7];
h q[4];
h q[15];
h q[6];
cx q[4],q[6];
h q[6];
cx q[11],q[9];
cx q[4],q[6];
h q[12];
s q[11];
h q[14];
h q[13];
s q[13];
h q[11];
