OPENQASM 2.0;
include "qelib1.inc";
qreg q[65];
h q[22];
h q[29];
h q[61];
h q[12];
cx q[18],q[49];
h q[18];
h q[27];
s q[51];
cx q[15],q[53];
h q[11];
h q[51];
cx q[52],q[22];
h q[38];
h q[52];
s q[57];
cx q[13],q[31];
s q[4];
cx q[17],q[47];
h q[57];
h q[6];
s q[12];
s q[8];
s q[26];
s q[4];
h q[36];
s q[7];
h q[64];
s q[16];
cx q[44],q[3];
s q[30];
cx q[60],q[43];
s q[38];
cx q[18],q[39];
s q[44];
h q[57];
h q[38];
h q[2];
h q[61];
h q[62];
h q[24];
h q[37];
h q[5];
s q[43];
h q[31];
h q[60];
cx q[20],q[15];
s q[35];
cx q[18],q[20];
h q[17];
cx q[48],q[13];
cx q[30],q[0];
h q[53];
s q[2];
cx q[56],q[10];
s q[15];
s q[23];
h q[7];
s q[35];
s q[7];
s q[59];
s q[27];
h q[40];
cx q[62],q[16];
h q[32];
h q[28];
h q[45];
cx q[5],q[18];
cx q[46],q[24];
s q[9];
s q[4];
cx q[32],q[54];
h q[17];
cx q[48],q[10];
h q[25];
cx q[58],q[26];
h q[32];
s q[0];
h q[54];
s q[4];
s q[2];
h q[22];
h q[52];
h q[16];
h q[0];
cx q[44],q[12];
s q[64];
h q[31];
s q[38];
s q[33];
s q[2];
cx q[49],q[11];
h q[53];
h q[56];
h q[46];
cx q[13],q[1];
cx q[50],q[38];
s q[63];
s q[37];
s q[29];
cx q[50],q[63];
s q[51];