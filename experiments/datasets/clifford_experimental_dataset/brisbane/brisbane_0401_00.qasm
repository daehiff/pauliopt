OPENQASM 2.0;
include "qelib1.inc";
qreg q[127];
cx q[20],q[73];
cx q[17],q[3];
h q[59];
s q[113];
s q[8];
s q[52];
s q[83];
cx q[59],q[71];
h q[7];
cx q[34],q[78];
h q[35];
s q[103];
s q[125];
s q[53];
s q[3];
s q[92];
cx q[17],q[90];
s q[73];
s q[99];
s q[94];
cx q[71],q[78];
cx q[123],q[120];
s q[39];
h q[79];
h q[81];
cx q[52],q[23];
s q[88];
h q[40];
h q[14];
h q[64];
h q[70];
h q[87];
h q[107];
cx q[10],q[115];
h q[7];
cx q[34],q[32];
cx q[4],q[106];
cx q[121],q[40];
cx q[72],q[71];
s q[32];
cx q[22],q[62];
h q[98];
s q[90];
cx q[64],q[99];
h q[46];
s q[2];
h q[4];
s q[118];
cx q[13],q[103];
cx q[8],q[79];
cx q[89],q[41];
h q[50];
cx q[95],q[103];
h q[51];
h q[95];
s q[100];
cx q[102],q[109];
cx q[42],q[28];
h q[31];
cx q[58],q[115];
s q[27];
s q[41];
h q[61];
h q[116];
s q[27];
s q[61];
cx q[121],q[118];
h q[102];
s q[96];
h q[120];
cx q[61],q[121];
h q[104];
cx q[102],q[69];
cx q[8],q[62];
h q[96];
cx q[105],q[43];
cx q[58],q[118];
h q[51];
h q[61];
h q[57];
s q[38];
s q[2];
h q[100];
h q[55];
h q[58];
h q[1];
s q[91];
s q[86];
h q[95];
h q[0];
cx q[125],q[1];
h q[43];
s q[31];
s q[31];
cx q[67],q[54];
cx q[55],q[123];
h q[126];
s q[23];
h q[115];
s q[69];
s q[10];
cx q[111],q[15];
h q[72];
cx q[69],q[80];
h q[2];
cx q[35],q[18];
s q[66];
cx q[18],q[20];
cx q[51],q[105];
h q[39];
cx q[81],q[104];
h q[10];
h q[88];
s q[22];
cx q[93],q[41];
cx q[6],q[124];
s q[111];
h q[115];
h q[1];
h q[47];
s q[11];
h q[36];
cx q[118],q[8];
h q[98];
cx q[47],q[80];
cx q[19],q[24];
s q[119];
h q[23];
cx q[112],q[71];
s q[103];
s q[98];
s q[88];
cx q[24],q[93];
s q[109];
s q[65];
s q[34];
h q[40];
h q[67];
h q[13];
h q[47];
cx q[66],q[16];
h q[47];
cx q[114],q[85];
cx q[21],q[110];
s q[37];
s q[108];
cx q[53],q[7];
cx q[26],q[98];
h q[29];
h q[27];
cx q[63],q[97];
h q[123];
h q[47];
cx q[3],q[35];
h q[16];
s q[92];
s q[116];
s q[98];
h q[36];
h q[112];
s q[52];
cx q[98],q[59];
h q[115];
cx q[124],q[84];
cx q[32],q[67];
s q[24];
s q[94];
s q[57];
cx q[103],q[45];
s q[31];
cx q[85],q[22];
s q[126];
cx q[105],q[1];
s q[16];
h q[8];
cx q[117],q[47];
cx q[92],q[41];
cx q[25],q[99];
s q[24];
h q[59];
s q[6];
h q[35];
h q[19];
h q[7];
s q[15];
s q[75];
cx q[86],q[14];
s q[65];
cx q[62],q[86];
cx q[24],q[58];
cx q[61],q[125];
s q[57];
s q[85];
h q[51];
s q[69];
cx q[53],q[107];
h q[96];
h q[59];
h q[102];
s q[108];
s q[46];
cx q[54],q[39];
h q[113];
s q[29];
cx q[16],q[120];
cx q[18],q[92];
s q[54];
s q[100];
s q[116];
s q[22];
cx q[8],q[12];
h q[0];
s q[121];
h q[110];
s q[95];
s q[117];
h q[103];
h q[116];
h q[15];
h q[102];
cx q[68],q[21];
h q[118];
cx q[75],q[25];
cx q[100],q[85];
h q[28];
s q[91];
h q[46];
h q[93];
s q[61];
h q[75];
s q[89];
h q[38];
h q[125];
s q[100];
cx q[121],q[9];
s q[68];
s q[51];
cx q[9],q[122];
cx q[57],q[96];
h q[119];
h q[108];
cx q[1],q[113];
cx q[83],q[23];
s q[34];
h q[32];
h q[50];
cx q[100],q[11];
cx q[64],q[32];
s q[42];
h q[12];
cx q[45],q[1];
h q[114];
s q[34];
cx q[80],q[90];
h q[25];
s q[89];
s q[33];
cx q[104],q[6];
s q[116];
s q[108];
cx q[28],q[120];
h q[20];
h q[120];
s q[100];
h q[23];
cx q[48],q[99];
s q[112];
cx q[95],q[23];
cx q[61],q[96];
h q[11];
h q[54];
h q[22];
h q[98];
h q[29];
h q[112];
s q[83];
h q[85];
h q[58];
cx q[48],q[100];
h q[104];
cx q[75],q[8];
cx q[27],q[78];
cx q[51],q[83];
s q[110];
h q[68];
cx q[11],q[25];
cx q[84],q[100];
h q[22];
h q[38];
h q[52];
s q[57];
cx q[13],q[95];
s q[4];
cx q[86],q[93];
cx q[74],q[17];
cx q[75],q[8];
s q[57];
h q[124];
s q[6];
s q[12];
s q[8];
s q[26];
s q[65];
h q[28];
h q[37];
cx q[7],q[112];
h q[64];
s q[115];
h q[126];
cx q[88],q[44];
s q[123];
cx q[18],q[61];
s q[38];
cx q[73],q[90];
cx q[38],q[126];
cx q[44],q[12];
s q[91];
s q[57];
h q[126];
h q[38];
h q[2];
h q[91];
s q[120];
cx q[123],q[112];
h q[55];
h q[37];
h q[5];
s q[43];
h q[31];
h q[60];
cx q[20],q[80];
s q[84];
cx q[123],q[100];
cx q[18],q[20];
h q[17];
cx q[104],q[48];
s q[14];
cx q[0],q[117];
s q[117];
cx q[15],q[87];
h q[74];
s q[95];
s q[75];
h q[7];
s q[91];
s q[7];
s q[59];
s q[27];
h q[40];
cx q[117],q[62];
h q[72];
h q[116];
h q[91];
h q[12];
s q[34];
h q[5];
s q[114];
h q[46];
h q[65];
s q[55];
s q[113];
cx q[108],q[4];
cx q[119],q[32];
s q[64];
s q[95];
cx q[48],q[10];
h q[25];
cx q[88],q[85];
cx q[26],q[49];
h q[32];
s q[104];
cx q[126],q[0];
h q[111];
cx q[5],q[92];
h q[68];
cx q[4],q[102];
cx q[52],q[22];
h q[52];
h q[73];
s q[82];
h q[84];
s q[72];
s q[0];
cx q[44],q[77];
s q[64];
h q[103];
s q[91];
cx q[71],q[38];
s q[121];
s q[53];
s q[2];
cx q[49],q[11];
h q[53];
h q[93];
s q[122];
h q[16];
cx q[46],q[22];
cx q[100],q[84];
s q[65];
cx q[50],q[103];
s q[63];
