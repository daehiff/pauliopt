OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
cx q[4],q[5];
h q[2];
h q[4];
h q[0];
h q[0];
cx q[0],q[2];
h q[4];
cx q[6],q[3];
s q[3];
s q[6];
cx q[0],q[3];
s q[3];
h q[5];
h q[2];
s q[3];
h q[4];
cx q[3],q[4];
s q[3];
h q[2];
s q[3];
s q[1];
cx q[6],q[0];
h q[6];
s q[5];
s q[0];
s q[0];
h q[3];
s q[1];
cx q[2],q[4];
s q[0];
h q[4];
h q[0];
cx q[5],q[2];
h q[3];
h q[4];
h q[5];
h q[5];
h q[5];
cx q[3],q[4];
h q[1];
cx q[4],q[5];
h q[5];
s q[2];
s q[5];
h q[0];
h q[0];
h q[0];
h q[2];
cx q[6],q[0];
cx q[4],q[2];
s q[2];
cx q[4],q[6];
h q[0];
cx q[3],q[5];
h q[4];
h q[0];
h q[1];
cx q[3],q[1];
h q[0];
cx q[5],q[6];
h q[6];
