OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
s q[10];
cx q[0],q[3];
h q[14];
s q[2];
h q[4];
s q[6];
cx q[13],q[6];
cx q[8],q[15];
cx q[9],q[10];
h q[2];
cx q[15],q[6];
h q[3];
h q[15];
s q[4];
cx q[6],q[13];
cx q[10],q[13];
h q[15];
cx q[10],q[2];
s q[11];
s q[9];
h q[13];