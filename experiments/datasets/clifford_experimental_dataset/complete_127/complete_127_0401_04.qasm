OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
cx q[75],q[92];
s q[89];
h q[27];
s q[61];
h q[84];
cx q[51],q[33];
h q[50];
cx q[81],q[57];
cx q[56],q[22];
s q[16];
h q[107];
cx q[93],q[14];
cx q[34],q[32];
h q[15];
s q[65];
cx q[39],q[120];
cx q[38],q[70];
h q[19];
h q[71];
cx q[86],q[62];
cx q[3],q[58];
h q[46];
s q[14];
cx q[20],q[99];
h q[116];
h q[45];
s q[99];
h q[38];
h q[98];
h q[78];
s q[114];
h q[5];
cx q[90],q[22];
cx q[90],q[98];
s q[37];
s q[99];
h q[71];
s q[34];
cx q[29],q[78];
cx q[83],q[113];
h q[39];
s q[25];
s q[86];
s q[47];
h q[82];
s q[50];
cx q[41],q[18];
h q[27];
h q[70];
s q[62];
h q[122];
cx q[120],q[124];
cx q[46],q[116];
s q[119];
s q[52];
h q[35];
h q[29];
h q[63];
h q[119];
h q[13];
cx q[52],q[99];
h q[58];
s q[93];
h q[67];
cx q[67],q[92];
s q[104];
h q[109];
s q[40];
cx q[126],q[54];
h q[61];
h q[47];
s q[11];
cx q[10],q[64];
s q[49];
h q[126];
h q[68];
h q[66];
h q[116];
h q[72];
h q[8];
cx q[80],q[98];
h q[7];
h q[86];
cx q[39],q[76];
s q[75];
s q[46];
cx q[104],q[18];
s q[51];
h q[21];
h q[50];
s q[19];
h q[28];
cx q[3],q[55];
cx q[91],q[27];
s q[75];
h q[66];
s q[107];
s q[87];
s q[90];
cx q[104],q[0];
cx q[37],q[5];
s q[24];
s q[16];
h q[11];
s q[89];
cx q[7],q[83];
s q[64];
h q[92];
h q[112];
cx q[39],q[13];
cx q[47],q[109];
h q[12];
h q[101];
h q[45];
h q[61];
cx q[19],q[124];
h q[89];
cx q[84],q[117];
h q[63];
s q[105];
cx q[53],q[126];
h q[8];
h q[48];
h q[16];
h q[50];
cx q[38],q[55];
h q[38];
h q[63];
cx q[109],q[18];
cx q[76],q[45];
h q[0];
cx q[75],q[71];
h q[116];
h q[11];
h q[111];
h q[44];
s q[102];
s q[68];
h q[90];
s q[86];
s q[108];
cx q[86],q[48];
cx q[83],q[81];
s q[65];
s q[32];
cx q[35],q[82];
h q[18];
cx q[27],q[40];
h q[96];
s q[114];
cx q[40],q[38];
s q[62];
h q[104];
s q[65];
cx q[74],q[77];
s q[13];
s q[50];
s q[72];
h q[38];
cx q[83],q[69];
cx q[116],q[1];
h q[37];
cx q[15],q[94];
s q[80];
h q[125];
cx q[106],q[64];
cx q[44],q[79];
s q[100];
h q[61];
h q[126];
cx q[24],q[22];
h q[37];
h q[53];
cx q[17],q[36];
cx q[44],q[119];
h q[52];
s q[121];
s q[59];
s q[31];
cx q[66],q[27];
cx q[40],q[63];
cx q[103],q[74];
s q[122];
h q[117];
h q[23];
cx q[33],q[56];
h q[6];
h q[55];
h q[64];
cx q[19],q[18];
h q[73];
s q[91];
cx q[42],q[111];
cx q[90],q[126];
cx q[48],q[113];
s q[63];
h q[7];
s q[47];
s q[64];
cx q[3],q[113];
h q[49];
h q[29];
cx q[66],q[75];
h q[3];
s q[26];
s q[117];
cx q[63],q[91];
s q[50];
s q[5];
cx q[124],q[14];
h q[29];
s q[100];
s q[124];
h q[41];
h q[34];
cx q[5],q[68];
s q[123];
h q[112];
h q[76];
cx q[46],q[126];
h q[108];
s q[67];
h q[36];
h q[48];
h q[33];
h q[85];
h q[122];
cx q[50],q[2];
s q[97];
h q[50];
s q[112];
h q[45];
cx q[107],q[20];
cx q[102],q[67];
cx q[0],q[92];
cx q[23],q[125];
cx q[12],q[111];
s q[10];
cx q[124],q[16];
cx q[123],q[81];
s q[120];
s q[18];
h q[102];
h q[7];
h q[16];
h q[35];
s q[18];
cx q[63],q[91];
h q[97];
cx q[53],q[44];
cx q[11],q[118];
h q[117];
cx q[102],q[91];
h q[80];
h q[45];
s q[11];
s q[16];
cx q[18],q[26];
s q[126];
cx q[121],q[99];
s q[74];
h q[61];
cx q[14],q[27];
cx q[63],q[121];
cx q[113],q[80];
cx q[52],q[78];
cx q[81],q[92];
cx q[46],q[111];
cx q[108],q[34];
cx q[72],q[71];
h q[80];
h q[112];
h q[95];
cx q[62],q[37];
cx q[58],q[69];
h q[46];
s q[60];
cx q[118],q[79];
s q[95];
s q[92];
s q[42];
cx q[53],q[9];
s q[63];
s q[20];
s q[40];
cx q[55],q[29];
h q[93];
cx q[34],q[18];
s q[7];
s q[112];
cx q[29],q[6];
cx q[64],q[35];
h q[36];
h q[14];
h q[12];
h q[59];
h q[126];
h q[35];
cx q[65],q[62];
cx q[8],q[34];
s q[77];
s q[94];
s q[74];
cx q[71],q[2];
s q[59];
h q[44];
s q[44];
h q[82];
cx q[101],q[15];
s q[39];
h q[118];
s q[105];
s q[59];
s q[65];
s q[12];
cx q[30],q[124];
s q[38];
cx q[7],q[87];
s q[119];
s q[54];
cx q[117],q[92];
cx q[52],q[38];
h q[16];
h q[29];
h q[12];
h q[18];
h q[18];
h q[24];
s q[30];
h q[54];
cx q[31],q[62];
s q[9];
s q[51];
s q[13];
s q[27];
s q[115];
h q[23];
cx q[21],q[30];
cx q[3],q[77];
cx q[86],q[7];
s q[11];
cx q[30],q[78];
s q[81];
s q[126];
cx q[104],q[75];
h q[78];
h q[94];
h q[121];
h q[25];
s q[95];
s q[37];
h q[86];
s q[7];
h q[5];
h q[43];
h q[63];
h q[27];
cx q[120],q[64];
cx q[18],q[8];
cx q[33],q[44];
s q[14];
cx q[12],q[83];
h q[69];
cx q[36],q[16];
s q[86];
s q[28];
h q[59];
cx q[88],q[120];
cx q[9],q[47];
cx q[84],q[20];
cx q[12],q[101];
h q[23];
cx q[16],q[68];
cx q[24],q[94];
h q[20];
cx q[82],q[11];
h q[41];
h q[100];
s q[47];
cx q[27],q[41];
cx q[46],q[112];
s q[77];
cx q[107],q[104];
cx q[104],q[12];
s q[100];
h q[99];
cx q[1],q[16];
h q[72];
s q[74];
s q[55];
s q[96];
h q[86];
h q[108];
s q[8];
cx q[35],q[17];
cx q[27],q[70];
h q[27];
s q[53];
h q[99];
h q[123];
cx q[98],q[12];