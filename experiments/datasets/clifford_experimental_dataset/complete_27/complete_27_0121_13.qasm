OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[7],q[1];
h q[14];
h q[18];
h q[18];
h q[10];
cx q[7],q[19];
cx q[26],q[7];
s q[26];
s q[18];
s q[9];
h q[6];
h q[0];
cx q[4],q[19];
h q[14];
cx q[6],q[26];
h q[9];
h q[4];
h q[25];
h q[21];
cx q[12],q[0];
cx q[22],q[0];
cx q[17],q[18];
h q[25];
h q[6];
h q[25];
h q[16];
s q[1];
h q[5];
s q[17];
s q[9];
s q[26];
h q[7];
s q[16];
cx q[25],q[5];
s q[21];
s q[5];
s q[6];
cx q[13],q[22];
cx q[22],q[15];
cx q[7],q[18];
h q[12];
h q[8];
s q[20];
cx q[26],q[8];
cx q[19],q[1];
h q[12];
s q[18];
s q[1];
cx q[5],q[2];
cx q[9],q[24];
cx q[5],q[23];
h q[18];
cx q[17],q[6];
h q[19];
h q[2];
cx q[24],q[3];
h q[0];
cx q[17],q[11];
h q[8];
s q[4];
h q[15];
cx q[21],q[3];
cx q[22],q[26];
s q[2];
s q[14];
cx q[9],q[8];
cx q[15],q[9];
h q[18];
s q[23];
cx q[1],q[6];
cx q[18],q[7];
s q[2];
s q[13];
h q[15];
h q[20];
cx q[21],q[26];
s q[8];
s q[22];
h q[24];
h q[20];
cx q[9],q[5];
cx q[11],q[3];
h q[16];
cx q[26],q[7];
s q[19];
cx q[11],q[6];
s q[23];
s q[21];
cx q[4],q[9];
h q[10];
h q[2];
cx q[16],q[1];
cx q[0],q[7];
s q[13];
h q[18];
cx q[10],q[23];
h q[1];
h q[26];
cx q[1],q[7];
h q[20];
s q[4];
h q[20];
h q[19];
cx q[5],q[20];
s q[24];
s q[24];
s q[5];
h q[24];
cx q[0],q[1];
cx q[4],q[13];
cx q[11],q[16];
h q[24];
cx q[2],q[12];
s q[4];
s q[21];
cx q[12],q[25];
cx q[21],q[2];
h q[19];
cx q[4],q[24];
s q[0];
s q[19];
