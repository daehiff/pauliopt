OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[10],q[21];
h q[22];
h q[26];
cx q[24],q[8];
s q[9];
s q[18];
h q[6];
h q[14];
s q[21];
s q[14];
h q[13];
s q[17];
s q[15];
h q[22];
cx q[10],q[15];
s q[19];
s q[5];
cx q[23],q[26];
s q[12];
h q[1];
h q[16];
s q[1];
s q[25];
cx q[7],q[12];
cx q[21],q[19];
h q[7];
s q[16];
cx q[15],q[13];
s q[10];
cx q[15],q[12];
cx q[14],q[24];
cx q[20],q[23];
cx q[8],q[2];
cx q[19],q[4];
s q[16];
cx q[22],q[25];
cx q[20],q[13];
cx q[20],q[23];
h q[16];
s q[18];
s q[10];
s q[17];
cx q[11],q[26];
s q[26];
s q[5];
s q[19];
cx q[4],q[20];
cx q[4],q[25];
cx q[7],q[11];
s q[7];
h q[23];
s q[6];
h q[3];
h q[12];
h q[25];
s q[3];
cx q[17],q[5];
cx q[23],q[24];
cx q[21],q[1];
cx q[22],q[10];
s q[4];
s q[9];
s q[23];
cx q[11],q[16];
s q[22];
h q[0];
cx q[9],q[25];
cx q[9],q[21];
s q[8];
s q[12];
h q[15];
s q[18];
h q[14];
s q[14];
cx q[22],q[6];
h q[16];
h q[7];
s q[26];
h q[17];
h q[15];
h q[11];