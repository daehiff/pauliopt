OPENQASM 2.0;
include "qelib1.inc";
qreg q[65];
h q[33];
h q[5];
cx q[24],q[33];
cx q[57],q[6];
h q[11];
cx q[6],q[36];
cx q[50],q[8];
cx q[7],q[33];
cx q[44],q[21];
cx q[3],q[40];
cx q[28],q[41];
h q[27];
h q[43];
s q[3];
s q[8];
cx q[4],q[56];
s q[14];
s q[16];
cx q[56],q[25];
h q[9];
cx q[44],q[18];
cx q[52],q[7];
cx q[27],q[10];
cx q[64],q[48];
cx q[51],q[37];
s q[59];
h q[46];
cx q[45],q[50];
s q[4];
h q[53];
h q[49];
s q[26];
s q[14];
h q[20];
s q[5];
s q[61];
s q[61];
s q[45];
h q[2];
s q[5];
h q[7];
h q[32];
h q[49];
cx q[27],q[44];
s q[40];
cx q[47],q[56];
cx q[61],q[56];
cx q[17],q[19];
cx q[54],q[37];
s q[25];
s q[3];
cx q[28],q[59];
h q[16];
cx q[42],q[53];
s q[33];
cx q[7],q[13];
s q[7];
s q[64];
cx q[38],q[16];
cx q[43],q[8];
h q[35];
h q[27];
s q[42];
h q[41];
s q[54];
cx q[21],q[37];
cx q[12],q[45];
s q[8];
s q[26];
s q[0];
h q[19];
h q[34];
cx q[16],q[18];
h q[52];
cx q[64],q[49];
h q[57];
s q[19];
cx q[9],q[59];
cx q[40],q[41];
h q[64];
h q[21];
s q[40];
h q[10];
cx q[20],q[35];
s q[18];
h q[59];
cx q[46],q[55];
cx q[25],q[40];
h q[20];
s q[3];
cx q[14],q[50];
h q[7];
h q[18];
cx q[45],q[27];
s q[14];
cx q[36],q[0];
cx q[23],q[27];
cx q[60],q[35];
s q[23];
s q[3];
cx q[33],q[57];