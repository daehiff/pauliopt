OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
h q[7];
cx q[0],q[8];
s q[7];
s q[2];
s q[8];
cx q[8],q[13];
s q[15];
s q[1];
s q[5];
cx q[15],q[12];
h q[3];
h q[3];
h q[13];
h q[15];
cx q[2],q[0];
h q[15];
cx q[2],q[6];
cx q[5],q[14];
s q[5];
s q[12];
cx q[5],q[8];
