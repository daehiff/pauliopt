OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
h q[15];
s q[3];
h q[10];
h q[14];
h q[11];
h q[0];
h q[7];
cx q[14],q[5];
cx q[12],q[3];
s q[13];
s q[8];
cx q[4],q[3];
s q[12];
h q[6];
cx q[5],q[4];
h q[10];
h q[0];
h q[2];
s q[11];
s q[3];
h q[1];
cx q[10],q[14];
s q[8];
s q[12];
h q[6];
cx q[7],q[15];
cx q[1],q[7];
s q[3];
cx q[15],q[13];
h q[3];
s q[9];
cx q[7],q[1];
h q[0];
cx q[7],q[13];
s q[4];
cx q[5],q[15];
s q[0];
cx q[6],q[10];
cx q[9],q[14];
s q[1];
h q[9];
