OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
h q[14];
cx q[1],q[14];
h q[7];
cx q[6],q[19];
s q[5];
cx q[6],q[19];
s q[19];
cx q[14],q[0];
s q[6];
cx q[3],q[6];
h q[0];
h q[4];
cx q[22],q[0];
s q[3];
s q[1];
s q[4];
cx q[9],q[0];
s q[5];
h q[14];
h q[23];
cx q[7],q[4];
h q[14];
h q[1];
cx q[18],q[19];
s q[10];
cx q[4],q[20];
cx q[9],q[7];
s q[5];
cx q[14],q[26];
s q[10];
s q[12];
s q[17];
h q[0];
cx q[7],q[11];
h q[23];
h q[7];
cx q[9],q[17];
cx q[25],q[8];
s q[18];
h q[13];
h q[21];
