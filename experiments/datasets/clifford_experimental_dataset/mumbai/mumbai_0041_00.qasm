OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
s q[25];
h q[25];
h q[1];
cx q[6],q[12];
h q[7];
cx q[2],q[14];
h q[3];
s q[7];
s q[5];
s q[9];
s q[17];
s q[11];
s q[9];
s q[3];
s q[15];
cx q[7],q[14];
cx q[24],q[7];
h q[15];
h q[17];
cx q[20],q[24];
s q[24];
h q[8];
h q[14];
h q[0];
h q[6];
h q[23];
h q[11];
cx q[10],q[19];
h q[7];
cx q[2],q[0];
cx q[4],q[10];
cx q[25],q[8];
cx q[8],q[7];
s q[0];
cx q[22],q[24];
h q[2];
s q[26];
cx q[0],q[3];
h q[14];
s q[2];
h q[4];