OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
s q[68];
s q[15];
h q[95];
s q[89];
s q[30];
cx q[9],q[56];
h q[75];
h q[91];
s q[122];
h q[22];
h q[57];
cx q[92],q[79];
s q[45];
s q[63];
cx q[19],q[51];
s q[38];
h q[49];
s q[74];
s q[103];
h q[30];
cx q[53],q[43];
s q[113];
cx q[121],q[112];
s q[61];
cx q[45],q[56];
cx q[117],q[88];
h q[2];
cx q[37],q[84];
s q[112];
s q[16];
cx q[9],q[99];
s q[44];
cx q[58],q[62];
cx q[48],q[100];
cx q[86],q[31];
s q[10];
s q[69];
cx q[104],q[61];
cx q[33],q[100];
h q[16];
s q[75];
h q[39];
cx q[66],q[82];
h q[114];
cx q[46],q[5];
s q[51];
cx q[38],q[49];
h q[11];
h q[9];
cx q[109],q[54];
h q[48];
h q[36];
cx q[24],q[6];
cx q[105],q[80];
h q[93];
cx q[96],q[70];
s q[23];
cx q[72],q[11];
cx q[62],q[68];
h q[12];
cx q[82],q[106];
h q[38];
h q[11];
s q[70];
cx q[30],q[93];
s q[12];
cx q[1],q[103];
cx q[56],q[75];
s q[102];
cx q[28],q[66];
h q[15];
cx q[76],q[36];
h q[20];
cx q[70],q[102];
h q[100];
s q[40];
s q[74];
s q[60];
s q[70];
h q[102];
cx q[69],q[107];
h q[106];
cx q[65],q[18];
s q[69];
h q[56];
h q[60];
s q[70];
h q[24];
h q[126];
h q[114];
s q[17];
s q[91];
cx q[96],q[64];
h q[100];
h q[51];
cx q[81],q[105];
s q[30];
cx q[16],q[55];
s q[92];
h q[25];
h q[16];
s q[124];
h q[1];
cx q[85],q[4];
s q[115];
h q[76];
s q[100];
cx q[88],q[25];
cx q[38],q[84];
h q[13];
h q[20];
s q[15];
s q[15];
h q[91];
cx q[88],q[18];
h q[65];
cx q[85],q[124];
s q[81];
s q[45];
h q[90];
h q[32];
h q[89];
h q[26];
h q[77];
cx q[110],q[115];
cx q[112],q[93];
cx q[47],q[63];
s q[11];
h q[42];
h q[39];
h q[84];
h q[86];
cx q[56],q[121];
s q[73];
s q[106];
s q[99];
cx q[118],q[22];
cx q[57],q[121];
h q[36];
s q[64];
s q[49];
s q[104];
h q[124];
cx q[89],q[71];
cx q[87],q[47];
s q[124];
s q[117];
cx q[105],q[64];
s q[35];
s q[75];
s q[10];
s q[113];
cx q[70],q[10];
cx q[32],q[12];
s q[40];
s q[55];
h q[30];
cx q[124],q[104];
s q[59];
s q[101];
cx q[2],q[43];
cx q[69],q[66];
cx q[74],q[14];
h q[115];
s q[37];
cx q[56],q[49];
s q[70];
cx q[17],q[82];
h q[87];
s q[103];
s q[42];
s q[70];
s q[93];
s q[4];
cx q[64],q[33];
h q[82];
s q[84];
h q[86];
h q[87];
cx q[11],q[22];
s q[0];
s q[125];
h q[23];
s q[35];
h q[74];
cx q[54],q[77];
h q[23];
s q[25];
s q[79];
s q[109];
h q[112];
cx q[50],q[34];
cx q[28],q[37];
h q[57];
s q[38];
s q[27];
h q[42];
cx q[115],q[9];
cx q[105],q[13];
cx q[41],q[94];
s q[31];
s q[114];
h q[55];
h q[121];
s q[1];
h q[29];
cx q[70],q[51];
cx q[8],q[22];
s q[109];
h q[0];
h q[83];
h q[82];
h q[67];
h q[104];
s q[17];
s q[42];
cx q[72],q[5];
cx q[47],q[126];
cx q[6],q[80];
h q[15];
h q[114];
h q[102];
cx q[117],q[36];
h q[80];
s q[85];
s q[71];
h q[124];
h q[65];
s q[110];
s q[106];
h q[73];
h q[3];
h q[88];
h q[17];
h q[122];
s q[47];
h q[57];
cx q[72],q[27];
cx q[120],q[102];
h q[9];
h q[41];
h q[8];
cx q[14],q[99];
h q[68];
h q[121];
s q[63];
h q[86];
cx q[55],q[22];
h q[90];
h q[5];
cx q[33],q[63];
h q[59];
h q[48];
h q[50];
s q[36];
cx q[97],q[40];
s q[92];
s q[13];
h q[78];
s q[35];
h q[27];
s q[79];
s q[75];
cx q[103],q[101];
s q[100];
h q[126];
s q[9];
s q[38];
s q[0];
cx q[20],q[65];
cx q[3],q[60];
h q[19];
s q[86];
h q[33];
s q[47];
h q[97];
cx q[60],q[9];
s q[83];
h q[76];
h q[4];
cx q[112],q[27];
cx q[22],q[112];
h q[106];
h q[112];
s q[12];
s q[10];
s q[95];
cx q[60],q[123];
cx q[80],q[111];
h q[9];
h q[76];
cx q[49],q[69];
s q[29];
cx q[16],q[75];
h q[1];
h q[0];
s q[121];
h q[2];
h q[85];
cx q[75],q[73];
s q[42];
s q[93];
s q[69];
s q[1];
s q[65];
s q[15];
h q[116];
s q[12];
cx q[52],q[43];
s q[45];
h q[113];
cx q[110],q[37];
cx q[10],q[117];
cx q[19],q[20];
h q[81];
h q[49];
s q[97];
s q[29];
h q[109];
cx q[30],q[42];
cx q[67],q[60];
h q[69];
s q[67];
s q[110];
cx q[16],q[19];
h q[50];
h q[69];
cx q[92],q[74];
s q[24];
cx q[70],q[25];
s q[88];
cx q[30],q[25];
s q[40];
cx q[46],q[12];
h q[65];
s q[15];
h q[27];
h q[34];
h q[76];
cx q[84],q[31];
s q[97];
cx q[40],q[112];
s q[101];
s q[39];
s q[27];
h q[104];
h q[104];
s q[54];
cx q[125],q[9];
cx q[74],q[13];
s q[11];
h q[16];
cx q[65],q[75];
h q[40];
cx q[28],q[17];
cx q[13],q[34];
h q[53];
cx q[57],q[104];
s q[57];
s q[29];
h q[62];
h q[18];
cx q[2],q[67];
cx q[85],q[80];
h q[13];
cx q[8],q[62];
s q[20];
h q[95];
cx q[19],q[35];
h q[46];
s q[74];
cx q[43],q[39];
s q[46];
s q[43];
cx q[84],q[51];
cx q[76],q[125];
h q[3];
h q[76];
s q[101];
s q[76];
cx q[91],q[57];
cx q[87],q[82];
h q[76];
h q[57];
h q[65];
s q[125];
s q[76];
s q[72];
h q[13];
h q[111];
cx q[95],q[32];
h q[23];
s q[61];
h q[61];
cx q[46],q[112];
s q[80];
cx q[7],q[116];
cx q[124],q[11];
s q[79];
cx q[116],q[48];
cx q[61],q[107];
