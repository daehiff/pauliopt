OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
h q[13];
h q[8];
cx q[7],q[14];
h q[1];
s q[8];
cx q[11],q[1];
h q[13];
cx q[2],q[7];
h q[0];
cx q[15],q[11];
h q[14];
h q[2];
s q[10];
cx q[12],q[11];
h q[4];
h q[12];
h q[9];
cx q[10],q[11];
s q[6];
s q[1];
s q[15];
cx q[11],q[1];
cx q[6],q[0];
h q[1];
s q[14];
s q[1];
cx q[3],q[1];
s q[10];
cx q[8],q[11];
h q[13];
cx q[2],q[3];
s q[4];
s q[11];
s q[3];
cx q[3],q[9];
s q[10];
s q[5];
cx q[6],q[3];
h q[5];
cx q[3],q[15];
h q[0];
cx q[8],q[15];
s q[4];
s q[9];
s q[11];
cx q[1],q[2];
s q[14];
s q[11];
s q[0];
cx q[13],q[8];
cx q[2],q[12];
cx q[6],q[7];
cx q[1],q[6];
s q[5];
h q[9];
cx q[13],q[1];
cx q[13],q[5];
s q[4];
s q[15];
cx q[2],q[6];
s q[9];
cx q[2],q[13];
h q[15];
h q[6];
h q[7];
s q[3];
s q[6];
h q[3];
s q[2];
h q[15];
s q[8];
cx q[12],q[2];
h q[7];
s q[3];
cx q[2],q[3];
h q[4];
s q[2];
cx q[7],q[12];
h q[7];
h q[3];
s q[13];
