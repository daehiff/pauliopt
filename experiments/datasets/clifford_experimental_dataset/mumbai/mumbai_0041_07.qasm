OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
h q[8];
cx q[0],q[3];
h q[24];
cx q[16],q[25];
h q[5];
h q[5];
s q[11];
h q[12];
h q[14];
h q[15];
s q[20];
cx q[4],q[3];
cx q[18],q[20];
h q[17];
cx q[8],q[17];
s q[14];
cx q[0],q[21];
s q[21];
cx q[15],q[23];
h q[10];
s q[15];
s q[11];
h q[7];
s q[3];
s q[7];
s q[17];
h q[8];
cx q[21],q[16];
h q[0];
h q[19];
h q[15];
h q[12];
s q[2];
h q[5];
s q[18];
h q[14];
h q[1];
s q[23];
s q[17];
cx q[12],q[4];
cx q[23],q[0];