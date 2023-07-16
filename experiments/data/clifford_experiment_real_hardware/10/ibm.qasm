OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg meas[5];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
rz(pi/2) q[3];
cx q[1],q[3];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[0],q[1];
rz(-pi) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[3];
cx q[1],q[3];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[3];
cx q[3],q[1];
cx q[1],q[3];
cx q[1],q[0];
rz(-pi) q[0];
sx q[0];
rz(pi/2) q[0];
x q[1];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[3];
cx q[3],q[1];
cx q[1],q[3];
cx q[2],q[1];
cx q[1],q[3];
cx q[3],q[1];
cx q[1],q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[3];
cx q[1],q[3];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi) q[2];
x q[3];
rz(pi/2) q[4];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[3],q[1];
cx q[1],q[3];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[3];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
rz(-pi) q[3];
x q[3];
barrier q[0],q[1],q[2],q[3],q[4];
x q[3];
rz(pi) q[3];
rz(pi) q[1];
sxdg q[1];
rz(pi/2) q[1];
cx q[1],q[3];
rz(-pi/2) q[1];
sxdg q[1];
rz(-pi/2) q[1];
cx q[1],q[3];
cx q[3],q[1];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[4];
rz(-pi/2) q[4];
x q[3];
rz(-pi) q[2];
rz(-pi/2) q[1];
sxdg q[1];
rz(pi) q[1];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[3];
cx q[4],q[3];
rz(-pi/2) q[4];
sxdg q[4];
rz(-pi/2) q[4];
cx q[1],q[3];
cx q[3],q[1];
cx q[1],q[3];
cx q[2],q[1];
cx q[1],q[3];
cx q[3],q[1];
cx q[1],q[3];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
x q[1];
rz(-pi/2) q[0];
sxdg q[0];
rz(pi) q[0];
cx q[1],q[0];
cx q[1],q[3];
cx q[3],q[1];
cx q[1],q[3];
rz(-pi/2) q[1];
sxdg q[1];
rz(-pi/2) q[1];
cx q[1],q[3];
cx q[4],q[3];
rz(-pi/2) q[4];
sxdg q[4];
rz(pi) q[4];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
rz(-pi/2) q[1];
sxdg q[1];
rz(pi) q[1];
cx q[1],q[3];
rz(-pi/2) q[3];
rz(-pi/2) q[2];
rz(-pi/2) q[1];
sxdg q[1];
rz(pi) q[1];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
