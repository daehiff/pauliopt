OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[11],q[26];
h q[23];
s q[0];
h q[2];
s q[8];
cx q[14],q[18];
s q[10];
h q[22];
s q[16];
h q[11];
cx q[14],q[2];
h q[19];
h q[15];
s q[1];
cx q[7],q[24];
cx q[6],q[5];
h q[19];
h q[7];
cx q[22],q[2];
s q[24];
cx q[5],q[15];
cx q[20],q[2];
h q[20];
h q[13];
s q[3];
h q[6];
h q[2];
h q[14];
s q[18];
h q[5];
cx q[26],q[22];
cx q[26],q[1];
s q[5];
s q[3];
h q[7];
s q[2];
cx q[13],q[20];
h q[24];
s q[25];
s q[22];
s q[15];
h q[18];
s q[18];
cx q[9],q[19];
h q[19];
h q[6];
s q[8];
cx q[22],q[25];
cx q[14],q[20];
s q[23];
s q[20];
h q[3];
h q[8];
h q[23];
h q[13];
cx q[20],q[2];
h q[26];
s q[8];
cx q[3],q[8];
s q[8];
h q[13];
s q[8];
cx q[22],q[15];
h q[16];
s q[11];
cx q[10],q[14];
s q[20];
cx q[20],q[4];
h q[2];
h q[20];
h q[8];
h q[8];
cx q[16],q[1];
h q[7];
h q[22];
cx q[7],q[12];
s q[11];
s q[14];
cx q[8],q[19];
s q[19];
h q[21];