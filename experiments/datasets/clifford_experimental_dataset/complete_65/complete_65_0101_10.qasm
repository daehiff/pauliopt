OPENQASM 2.0;
include "qelib1.inc";
qreg q[65];
cx q[48],q[35];
s q[40];
h q[61];
cx q[9],q[21];
cx q[1],q[54];
h q[52];
cx q[15],q[32];
s q[10];
s q[14];
h q[39];
s q[54];
h q[24];
cx q[21],q[52];
s q[43];
cx q[45],q[49];
h q[31];
s q[38];
h q[11];
h q[46];
s q[20];
h q[20];
s q[7];
s q[6];
cx q[21],q[43];
h q[21];
h q[12];
s q[1];
h q[51];
cx q[49],q[30];
cx q[22],q[24];
h q[21];
cx q[36],q[61];
h q[55];
cx q[62],q[53];
cx q[22],q[29];
s q[38];
cx q[24],q[64];
h q[7];
s q[57];
h q[4];
cx q[64],q[3];
cx q[63],q[41];
s q[9];
h q[45];
cx q[26],q[1];
h q[12];
h q[35];
s q[57];
s q[11];
h q[13];
h q[33];
h q[19];
cx q[15],q[51];
cx q[55],q[18];
h q[57];
cx q[3],q[20];
s q[23];
s q[25];
h q[53];
cx q[20],q[9];
s q[45];
h q[44];
h q[3];
h q[13];
cx q[57],q[54];
s q[23];
s q[29];
s q[18];
cx q[59],q[60];
cx q[48],q[63];
s q[21];
h q[29];
s q[30];
cx q[47],q[44];
s q[54];
h q[54];
h q[17];
cx q[46],q[10];
h q[44];
h q[16];
s q[3];
cx q[14],q[5];
cx q[8],q[26];
h q[60];
s q[49];
cx q[50],q[53];
cx q[41],q[34];
s q[25];
cx q[63],q[36];
h q[20];
s q[63];
h q[3];
s q[38];
s q[34];
h q[23];
h q[40];
s q[45];
h q[30];
h q[35];
h q[29];
s q[29];