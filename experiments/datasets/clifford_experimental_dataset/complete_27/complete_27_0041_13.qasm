OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[22],q[0];
s q[4];
h q[25];
s q[3];
s q[22];
h q[0];
cx q[21],q[23];
cx q[7],q[15];
h q[14];
h q[20];
h q[9];
h q[20];
s q[8];
cx q[19],q[25];
s q[10];
s q[1];
cx q[17],q[1];
cx q[16],q[10];
cx q[11],q[19];
s q[23];
h q[6];
s q[20];
h q[16];
h q[10];
s q[25];
s q[20];
h q[6];
h q[19];
h q[16];
s q[21];
cx q[7],q[17];
cx q[0],q[13];
s q[22];
cx q[26],q[23];
cx q[16],q[12];
h q[15];
cx q[17],q[0];
s q[18];
cx q[17],q[1];
s q[16];
s q[5];
