OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[3],q[7];
cx q[14],q[5];
h q[9];
s q[13];
h q[0];
cx q[4],q[6];
cx q[7],q[9];
s q[10];
cx q[14],q[3];
s q[6];
s q[0];
cx q[0],q[11];
cx q[6],q[1];
h q[14];
cx q[9],q[2];
s q[15];
cx q[4],q[14];
s q[10];
h q[15];
cx q[13],q[11];
cx q[4],q[6];
h q[4];
s q[7];
h q[2];
s q[12];
h q[7];
cx q[10],q[3];
cx q[2],q[4];
h q[12];
h q[3];
s q[3];
h q[13];
h q[2];
h q[9];
cx q[0],q[12];
s q[7];
h q[10];
s q[14];
s q[10];
s q[5];
h q[14];
h q[2];
cx q[12],q[1];
cx q[13],q[6];
h q[3];
s q[4];
cx q[12],q[15];
s q[7];
s q[0];
h q[13];
h q[3];
h q[9];
cx q[8],q[13];
s q[15];
cx q[14],q[0];
h q[10];
h q[2];
h q[0];
s q[2];
cx q[8],q[5];
cx q[3],q[9];
h q[5];
cx q[10],q[0];
h q[3];
s q[3];
cx q[10],q[1];
cx q[9],q[11];
s q[9];
h q[3];
cx q[0],q[2];
s q[9];
cx q[11],q[0];
s q[6];
s q[9];
s q[1];
h q[9];
s q[8];
h q[9];
h q[15];
h q[15];
h q[8];
s q[14];
cx q[1],q[3];
h q[2];
s q[7];
cx q[9],q[6];
h q[0];
h q[3];
cx q[7],q[8];
cx q[2],q[5];
cx q[7],q[2];
s q[8];
cx q[11],q[1];
s q[1];
h q[11];
s q[11];
h q[5];
cx q[11],q[13];
h q[10];
s q[0];
h q[11];
h q[12];
h q[14];
h q[4];
cx q[9],q[13];
cx q[10],q[9];
h q[12];
s q[2];
cx q[6],q[1];
s q[7];
cx q[12],q[8];
cx q[8],q[2];
cx q[15],q[4];
h q[13];
h q[0];
h q[0];
cx q[3],q[13];
cx q[11],q[10];
s q[11];
cx q[3],q[0];
s q[1];
s q[10];
cx q[14],q[5];
cx q[11],q[10];
s q[15];
s q[4];
s q[9];
s q[8];
h q[12];
s q[12];
s q[2];
h q[10];
s q[6];
h q[3];
h q[9];
h q[14];
cx q[12],q[0];
s q[14];
h q[2];
s q[15];
cx q[5],q[12];
cx q[9],q[8];
h q[9];
cx q[15],q[10];
cx q[11],q[0];
h q[6];
h q[0];
h q[7];
s q[1];
h q[8];
h q[14];
s q[10];
h q[7];
h q[14];
h q[9];
s q[4];
s q[15];
h q[12];
h q[5];
cx q[5],q[1];
cx q[15],q[1];
h q[5];
h q[7];
h q[7];
s q[11];
cx q[5],q[12];
s q[7];
s q[4];
h q[11];
cx q[6],q[3];
s q[6];
s q[4];
s q[3];
s q[1];
s q[3];
cx q[5],q[9];
s q[12];
cx q[13],q[0];
s q[13];
cx q[8],q[13];
h q[13];
cx q[11],q[9];
cx q[11],q[10];
h q[3];
cx q[3],q[5];
s q[15];
s q[3];
cx q[3],q[4];
h q[8];
s q[14];
h q[8];
cx q[1],q[12];
h q[4];
cx q[6],q[8];
h q[11];
h q[11];
cx q[2],q[13];
cx q[8],q[6];
h q[8];
cx q[13],q[12];
s q[1];
cx q[11],q[0];
s q[4];
h q[13];
s q[0];
s q[9];
h q[13];
s q[0];
s q[15];
cx q[12],q[15];
h q[11];
cx q[11],q[10];
s q[12];
h q[5];
s q[2];
h q[13];
s q[10];
cx q[3],q[14];
cx q[8],q[15];
h q[15];
s q[9];
cx q[15],q[13];
cx q[10],q[2];
cx q[12],q[11];
h q[9];
cx q[9],q[15];
s q[11];
h q[11];
s q[3];
cx q[6],q[14];
s q[3];
cx q[9],q[8];
cx q[14],q[7];
h q[5];
h q[7];
cx q[12],q[5];
s q[2];
cx q[12],q[5];
h q[6];
h q[13];
cx q[11],q[8];
h q[5];
s q[12];
s q[7];
cx q[0],q[4];
h q[2];
s q[6];
cx q[4],q[11];
s q[4];
s q[10];
cx q[5],q[9];
cx q[7],q[11];
cx q[15],q[8];
h q[6];
h q[7];
s q[10];
s q[3];
h q[12];
s q[15];
h q[6];
s q[1];
cx q[15],q[0];
s q[13];
cx q[8],q[3];
s q[12];
s q[0];
s q[4];
h q[1];
h q[0];
cx q[10],q[15];
cx q[8],q[1];
s q[6];
s q[13];
h q[8];
s q[8];
cx q[6],q[5];
s q[7];
cx q[9],q[5];
h q[4];
cx q[2],q[8];
cx q[14],q[7];