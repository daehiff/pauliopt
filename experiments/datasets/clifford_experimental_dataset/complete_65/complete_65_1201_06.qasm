OPENQASM 2.0;
include "qelib1.inc";
qreg q[65];
h q[3];
s q[53];
h q[39];
h q[30];
s q[24];
cx q[26],q[62];
s q[48];
cx q[23],q[21];
cx q[37],q[9];
s q[23];
s q[47];
h q[25];
h q[37];
s q[45];
cx q[57],q[64];
s q[30];
s q[54];
cx q[42],q[52];
cx q[14],q[27];
cx q[39],q[30];
s q[34];
s q[46];
s q[23];
s q[48];
h q[15];
s q[24];
h q[38];
cx q[1],q[58];
h q[41];
cx q[45],q[5];
h q[52];
h q[48];
cx q[41],q[11];
h q[54];
cx q[16],q[39];
cx q[6],q[64];
cx q[60],q[2];
s q[2];
s q[41];
cx q[23],q[13];
s q[58];
s q[19];
cx q[20],q[22];
s q[42];
cx q[16],q[12];
h q[5];
cx q[12],q[21];
h q[62];
cx q[14],q[5];
cx q[42],q[29];
cx q[51],q[7];
h q[13];
h q[5];
h q[1];
h q[11];
h q[1];
cx q[17],q[21];
s q[30];
h q[53];
cx q[10],q[62];
cx q[55],q[58];
cx q[36],q[18];
s q[33];
s q[43];
s q[63];
s q[5];
s q[0];
h q[17];
cx q[0],q[59];
h q[38];
h q[7];
s q[28];
s q[0];
cx q[8],q[58];
cx q[22],q[64];
cx q[28],q[36];
cx q[42],q[55];
cx q[13],q[48];
s q[40];
cx q[58],q[51];
cx q[25],q[19];
s q[61];
cx q[19],q[56];
h q[2];
s q[44];
h q[1];
h q[28];
h q[9];
cx q[33],q[15];
h q[31];
s q[0];
s q[34];
s q[26];
h q[8];
cx q[26],q[30];
h q[2];
s q[31];
h q[21];
h q[14];
cx q[61],q[10];
cx q[57],q[14];
cx q[49],q[50];
h q[9];
s q[13];
cx q[59],q[40];
cx q[20],q[38];
h q[8];
s q[9];
s q[51];
cx q[44],q[41];
s q[50];
s q[25];
cx q[54],q[13];
h q[9];
h q[40];
s q[19];
cx q[18],q[49];
cx q[2],q[41];
h q[56];
s q[11];
s q[43];
cx q[63],q[27];
h q[23];
cx q[0],q[59];
cx q[5],q[24];
h q[18];
h q[31];
s q[6];
h q[31];
s q[45];
s q[59];
cx q[40],q[17];
cx q[14],q[15];
h q[46];
cx q[44],q[21];
h q[44];
cx q[13],q[35];
s q[1];
cx q[51],q[52];
s q[44];
s q[51];
h q[10];
cx q[21],q[8];
h q[62];
h q[36];
h q[64];
cx q[26],q[2];
cx q[35],q[25];
s q[50];
s q[50];
s q[29];
cx q[35],q[38];
cx q[15],q[26];
s q[48];
s q[28];
h q[6];
s q[53];
h q[36];
h q[59];
h q[47];
s q[48];
h q[51];
h q[7];
h q[31];
cx q[24],q[27];
h q[44];
cx q[12],q[40];
s q[58];
cx q[56],q[17];
cx q[23],q[10];
h q[32];
h q[60];
cx q[57],q[26];
s q[58];
h q[8];
cx q[25],q[43];
s q[12];
cx q[51],q[22];
cx q[45],q[33];
s q[15];
h q[8];
cx q[40],q[56];
cx q[6],q[25];
h q[45];
h q[63];
h q[31];
s q[26];
h q[31];
h q[55];
h q[31];
cx q[59],q[36];
s q[30];
h q[28];
s q[42];
h q[24];
h q[58];
cx q[35],q[50];
cx q[50],q[35];
cx q[53],q[46];
cx q[3],q[9];
cx q[52],q[25];
h q[55];
s q[63];
cx q[27],q[53];
s q[45];
s q[60];
h q[51];
h q[51];
cx q[24],q[45];
cx q[47],q[51];
s q[63];
cx q[28],q[56];
h q[45];
h q[48];
cx q[61],q[43];
s q[14];
s q[56];
cx q[4],q[44];
cx q[36],q[42];
cx q[48],q[30];
h q[36];
cx q[53],q[11];
cx q[1],q[27];
h q[31];
s q[20];
h q[27];
h q[20];
s q[42];
s q[34];
h q[24];
h q[45];
s q[3];
cx q[16],q[23];
h q[0];
s q[39];
h q[23];
h q[39];
cx q[4],q[18];
s q[58];
cx q[40],q[6];
s q[38];
cx q[62],q[50];
cx q[55],q[30];
s q[10];
h q[41];
cx q[30],q[10];
s q[57];
h q[31];
s q[47];
s q[62];
h q[14];
h q[0];
s q[34];
s q[64];
h q[37];
s q[21];
s q[63];
h q[62];
cx q[55],q[49];
s q[58];
cx q[28],q[23];
h q[18];
s q[59];
s q[32];
cx q[60],q[44];
cx q[12],q[60];
cx q[56],q[35];
cx q[64],q[26];
cx q[12],q[11];
s q[0];
s q[50];
h q[1];
s q[52];
cx q[27],q[39];
cx q[40],q[16];
h q[38];
h q[36];
h q[49];
s q[56];
cx q[12],q[49];
s q[44];
cx q[4],q[24];
h q[1];
h q[19];
s q[21];
h q[33];
cx q[34],q[27];
cx q[40],q[58];
h q[53];
h q[10];
h q[37];
h q[52];
cx q[39],q[33];
h q[45];
s q[10];
s q[33];
h q[14];
s q[30];
cx q[62],q[47];
s q[9];
h q[18];
h q[9];
s q[4];
cx q[45],q[64];
s q[45];
s q[59];
cx q[19],q[23];
h q[30];
cx q[28],q[0];
cx q[9],q[23];
cx q[54],q[1];
cx q[26],q[45];
cx q[40],q[18];
h q[32];
h q[7];
h q[14];
s q[19];
h q[18];
cx q[49],q[8];
h q[13];
h q[34];
s q[38];
h q[20];
h q[47];
cx q[45],q[39];
cx q[60],q[27];
cx q[59],q[34];
h q[0];
s q[53];
s q[55];
h q[19];
cx q[14],q[15];
cx q[13],q[12];
s q[40];
h q[3];
s q[12];
s q[34];
s q[4];
cx q[32],q[41];
s q[36];
h q[34];
cx q[57],q[51];
cx q[50],q[25];
s q[4];
h q[43];
s q[36];
h q[45];
cx q[38],q[7];
h q[5];
s q[0];
cx q[33],q[54];
cx q[55],q[54];
h q[52];
cx q[5],q[18];
cx q[44],q[58];
s q[40];
h q[44];
s q[53];
cx q[12],q[43];
h q[35];
h q[19];
cx q[49],q[55];
s q[2];
s q[48];
cx q[35],q[14];
s q[12];
cx q[43],q[6];
s q[33];
s q[5];
h q[54];
s q[44];
cx q[35],q[39];
h q[31];
h q[56];
s q[28];
cx q[30],q[51];
h q[27];
h q[55];
h q[47];
cx q[48],q[13];
s q[52];
cx q[27],q[46];
s q[20];
h q[21];
cx q[14],q[62];
s q[55];
s q[63];
s q[30];
cx q[9],q[22];
cx q[2],q[9];
s q[43];
cx q[26],q[18];
s q[13];
s q[51];
s q[24];
cx q[13],q[23];
cx q[21],q[54];
s q[16];
s q[42];
cx q[52],q[64];
s q[22];
cx q[57],q[16];
cx q[19],q[44];
s q[3];
h q[26];
cx q[38],q[31];
h q[40];
cx q[5],q[30];
s q[11];
cx q[22],q[18];
cx q[39],q[53];
cx q[28],q[32];
h q[32];
h q[1];
cx q[48],q[17];
h q[59];
h q[57];
cx q[15],q[6];
s q[38];
cx q[8],q[53];
cx q[21],q[2];
h q[51];
s q[6];
s q[40];
s q[41];
h q[24];
h q[11];
h q[13];
h q[50];
h q[5];
h q[29];
s q[0];
h q[0];
h q[44];
cx q[7],q[28];
s q[26];
s q[23];
cx q[12],q[62];
h q[43];
s q[14];
s q[15];
h q[36];
s q[60];
cx q[22],q[39];
h q[39];
s q[52];
cx q[35],q[47];
s q[30];
h q[5];
h q[39];
h q[61];
s q[8];
cx q[2],q[3];
cx q[46],q[22];
s q[4];
s q[11];
cx q[12],q[22];
h q[33];
h q[57];
s q[48];
s q[19];
h q[47];
cx q[51],q[53];
cx q[46],q[52];
s q[54];
h q[30];
h q[20];
h q[56];
cx q[2],q[60];
h q[37];
cx q[28],q[12];
s q[12];
h q[0];
s q[48];
cx q[8],q[63];
h q[19];
h q[55];
h q[7];
s q[43];
cx q[56],q[52];
cx q[33],q[60];
h q[62];
cx q[20],q[8];
cx q[19],q[41];
h q[43];
h q[0];
cx q[27],q[48];
s q[34];
cx q[49],q[50];
s q[12];
h q[51];
s q[56];
h q[29];
cx q[38],q[6];
cx q[58],q[32];
s q[38];
s q[17];
h q[43];
h q[50];
cx q[8],q[41];
s q[52];
s q[63];
cx q[31],q[53];
h q[44];
h q[26];
cx q[17],q[51];
s q[53];
s q[55];
cx q[60],q[11];
cx q[64],q[3];
h q[61];
s q[22];
cx q[21],q[41];
s q[45];
cx q[7],q[62];
h q[39];
h q[24];
cx q[40],q[60];
s q[34];
h q[57];
s q[9];
cx q[55],q[31];
h q[46];
s q[0];
s q[26];
h q[39];
s q[58];
s q[59];
h q[45];
h q[24];
h q[33];
s q[15];
cx q[11],q[15];
s q[57];
s q[49];
s q[9];
cx q[10],q[38];
cx q[59],q[5];
s q[12];
h q[48];
cx q[61],q[20];
s q[40];
cx q[54],q[47];
h q[32];
cx q[34],q[10];
h q[42];
cx q[62],q[61];
s q[14];
h q[47];
s q[14];
h q[41];
cx q[45],q[31];
s q[2];
cx q[12],q[57];
cx q[21],q[53];
cx q[60],q[56];
s q[49];
h q[42];
cx q[37],q[20];
h q[27];
cx q[28],q[5];
h q[56];
s q[18];
s q[9];
cx q[58],q[18];
cx q[20],q[50];
s q[24];
h q[55];
h q[48];
cx q[58],q[51];
s q[42];
s q[46];
cx q[9],q[44];
s q[37];
h q[21];
cx q[45],q[59];
s q[39];
cx q[54],q[26];
s q[43];
cx q[13],q[35];
cx q[44],q[60];
h q[0];
cx q[56],q[6];
h q[47];
h q[29];
h q[49];
h q[32];
cx q[35],q[9];
h q[12];
s q[35];
h q[13];
cx q[42],q[5];
cx q[39],q[24];
h q[37];
cx q[61],q[2];
s q[50];
s q[17];
h q[9];
s q[21];
h q[50];
h q[64];
cx q[0],q[57];
s q[54];
h q[41];
cx q[34],q[27];
h q[12];
h q[16];
h q[57];
h q[27];
h q[35];
s q[32];
cx q[14],q[28];
cx q[38],q[39];
h q[55];
s q[35];
s q[57];
cx q[24],q[50];
s q[25];
cx q[26],q[62];
s q[35];
h q[64];
cx q[11],q[42];
s q[41];
h q[60];
h q[44];
h q[29];
cx q[62],q[56];
cx q[59],q[10];
s q[16];
h q[37];
cx q[57],q[15];
h q[25];
s q[49];
h q[4];
cx q[42],q[32];
h q[40];
h q[25];
s q[7];
s q[41];
cx q[52],q[20];
cx q[17],q[18];
cx q[30],q[50];
cx q[43],q[35];
h q[38];
h q[61];
cx q[53],q[0];
cx q[13],q[62];
cx q[45],q[7];
s q[53];
h q[33];
s q[35];
s q[48];
s q[3];
h q[54];
h q[21];
cx q[26],q[57];
h q[33];
s q[60];
h q[54];
s q[22];
s q[47];
h q[21];
h q[14];
s q[21];
cx q[35],q[51];
h q[22];
cx q[48],q[28];
h q[11];
h q[59];
cx q[39],q[23];
s q[35];
cx q[25],q[40];
h q[61];
s q[54];
cx q[38],q[30];
h q[22];
h q[24];
s q[49];
h q[20];
s q[54];
s q[41];
s q[40];
h q[32];
s q[14];
cx q[14],q[53];
h q[48];
h q[16];
s q[42];
h q[26];
cx q[8],q[41];
s q[34];
h q[29];
s q[49];
h q[2];
cx q[13],q[52];
h q[27];
cx q[52],q[18];
cx q[9],q[25];
s q[13];
h q[4];
s q[1];
h q[29];
cx q[31],q[37];
cx q[60],q[11];
cx q[12],q[7];
cx q[25],q[43];
cx q[37],q[19];
s q[15];
cx q[31],q[28];
s q[12];
s q[22];
s q[52];
h q[24];
cx q[25],q[0];
cx q[33],q[14];
s q[31];
s q[36];
cx q[57],q[8];
h q[58];
s q[17];
cx q[51],q[4];
h q[2];
h q[1];
s q[60];
h q[47];
h q[34];
cx q[56],q[14];
h q[31];
s q[48];
h q[39];
cx q[56],q[22];
s q[19];
h q[11];
s q[30];
h q[40];
s q[55];
s q[60];
cx q[32],q[52];
s q[25];
h q[23];
s q[34];
h q[6];
cx q[26],q[22];
s q[59];
cx q[22],q[25];
cx q[3],q[28];
h q[30];
s q[44];
cx q[36],q[7];
cx q[7],q[62];
cx q[4],q[3];
h q[32];
h q[35];
cx q[25],q[64];
h q[7];
h q[23];
cx q[48],q[39];
h q[23];
h q[9];
s q[59];
h q[32];
h q[5];
s q[51];
s q[22];
cx q[62],q[64];
s q[51];
cx q[6],q[58];
h q[10];
h q[50];
s q[31];
h q[30];
cx q[11],q[10];
h q[56];
s q[24];
cx q[44],q[17];
s q[19];
cx q[25],q[2];
s q[23];
s q[37];
s q[54];
h q[47];
cx q[64],q[30];
h q[54];
cx q[38],q[56];
h q[61];
cx q[3],q[26];
h q[47];
cx q[57],q[4];
s q[40];
cx q[59],q[14];
cx q[24],q[59];
h q[34];
s q[27];
cx q[52],q[27];
s q[7];
s q[40];
s q[15];
s q[60];
cx q[60],q[51];
s q[3];
h q[8];
cx q[27],q[13];
s q[30];
cx q[22],q[28];
cx q[13],q[63];
s q[49];
h q[31];
s q[49];
h q[30];
s q[53];
h q[50];
cx q[57],q[4];
h q[6];
cx q[41],q[56];
s q[28];
cx q[36],q[25];
s q[32];
s q[62];
h q[34];
h q[61];
s q[38];
cx q[25],q[3];
s q[28];
cx q[10],q[22];
s q[56];
h q[59];
cx q[62],q[0];
h q[18];
cx q[62],q[23];
h q[29];
s q[17];
s q[20];
h q[54];
h q[9];
cx q[38],q[18];
h q[59];
s q[50];
cx q[8],q[23];
s q[34];
s q[12];
s q[50];
h q[7];
cx q[13],q[8];
h q[35];
s q[25];
cx q[6],q[40];
h q[29];
h q[20];
cx q[43],q[37];
h q[51];
s q[30];
s q[43];
s q[32];
s q[53];
h q[55];
s q[30];
s q[13];
h q[16];
h q[51];
h q[24];
cx q[52],q[12];
h q[49];
cx q[22],q[19];
s q[31];
h q[37];
h q[60];
h q[28];
h q[27];
s q[52];
s q[14];
s q[33];
cx q[49],q[3];
s q[50];
cx q[11],q[61];
cx q[28],q[52];
cx q[37],q[50];
h q[56];
s q[54];
h q[7];
h q[43];
h q[25];
h q[36];
s q[4];
h q[10];
h q[15];
h q[7];
h q[63];
cx q[7],q[28];
h q[33];
s q[8];
cx q[28],q[25];
s q[40];
cx q[25],q[20];
cx q[8],q[64];
h q[4];
h q[13];
h q[29];
s q[10];
cx q[37],q[0];
h q[30];
cx q[47],q[24];
h q[7];
s q[2];
h q[13];
cx q[53],q[41];
h q[19];
cx q[22],q[56];
s q[1];
s q[2];
h q[35];
s q[22];
s q[2];
cx q[15],q[31];
s q[55];
s q[0];
h q[54];
h q[55];
cx q[1],q[10];
s q[52];
s q[41];
h q[39];
cx q[4],q[13];
cx q[0],q[50];
s q[10];
cx q[43],q[45];
h q[1];
cx q[20],q[58];
h q[22];
h q[0];
cx q[23],q[44];
h q[54];
h q[20];
cx q[39],q[37];
h q[1];
cx q[25],q[36];
s q[0];
cx q[32],q[42];
s q[24];
h q[2];
cx q[18],q[40];
s q[55];
h q[54];
s q[36];
cx q[16],q[6];
h q[63];
h q[61];
h q[18];
cx q[50],q[51];
h q[44];
h q[62];
h q[0];
s q[37];
h q[52];
cx q[32],q[30];
s q[56];
cx q[14],q[47];
s q[38];
cx q[22],q[60];
h q[60];
cx q[51],q[13];
cx q[15],q[27];
cx q[26],q[59];
cx q[51],q[55];
cx q[17],q[14];
h q[38];
cx q[46],q[39];
cx q[16],q[24];
s q[59];
h q[9];
h q[7];
cx q[27],q[3];
cx q[46],q[4];
h q[45];
s q[39];
h q[5];
s q[31];
s q[6];
cx q[44],q[62];
h q[44];
s q[5];
cx q[9],q[58];
s q[12];
h q[31];
h q[45];
cx q[10],q[31];
h q[21];
s q[17];
cx q[12],q[48];
s q[22];
h q[10];
cx q[29],q[41];
h q[33];
s q[35];
h q[51];
cx q[0],q[6];
s q[59];
s q[43];
h q[30];
s q[14];
cx q[6],q[32];
cx q[59],q[22];
h q[5];
h q[2];
cx q[43],q[58];
h q[27];
cx q[55],q[56];
h q[15];
s q[52];
s q[21];
h q[47];
cx q[28],q[52];
cx q[64],q[59];
s q[20];
s q[40];
cx q[57],q[46];
h q[38];
cx q[15],q[61];
h q[9];
h q[15];
cx q[22],q[61];
s q[42];
cx q[17],q[23];
h q[11];
h q[9];
cx q[40],q[11];
cx q[51],q[31];
cx q[19],q[10];
s q[30];
s q[51];
s q[57];
cx q[55],q[10];
h q[43];
h q[3];
h q[33];
cx q[20],q[7];
cx q[45],q[41];
cx q[32],q[26];
cx q[33],q[31];
cx q[56],q[58];
h q[38];
h q[50];
h q[7];
s q[43];
h q[1];
h q[62];
cx q[52],q[7];
cx q[50],q[11];
cx q[3],q[24];
s q[32];
cx q[28],q[51];
cx q[17],q[15];
cx q[4],q[36];
h q[60];
h q[36];
cx q[35],q[0];
h q[56];
cx q[51],q[19];
s q[11];
s q[52];
h q[48];
s q[9];
s q[64];
s q[33];
h q[18];
s q[63];
cx q[27],q[8];
cx q[14],q[19];
cx q[51],q[19];
s q[29];
h q[51];
s q[46];
s q[22];
s q[38];
s q[35];
h q[2];
s q[31];
cx q[16],q[39];
cx q[26],q[23];
h q[37];
cx q[40],q[56];
s q[4];
h q[23];
cx q[5],q[25];
s q[16];
h q[47];
cx q[21],q[7];
cx q[49],q[28];
cx q[24],q[1];
cx q[50],q[5];
s q[9];
s q[37];
cx q[8],q[16];
s q[37];
h q[9];
cx q[53],q[59];
s q[20];
s q[24];
cx q[55],q[22];
s q[19];
h q[7];
s q[35];
cx q[8],q[20];
s q[60];
s q[25];
cx q[29],q[4];
s q[44];
s q[22];
cx q[51],q[50];
cx q[46],q[29];
cx q[47],q[61];
s q[50];
cx q[33],q[21];
s q[49];
h q[3];
cx q[26],q[49];
h q[12];
s q[11];
s q[4];
cx q[52],q[59];
h q[39];
h q[50];
cx q[7],q[46];
cx q[45],q[15];
s q[39];
cx q[1],q[30];
s q[41];
h q[64];
s q[1];
s q[5];
cx q[24],q[8];
cx q[36],q[2];
s q[35];
h q[19];
h q[2];
s q[63];
cx q[14],q[50];
h q[57];
cx q[19],q[4];
h q[31];
cx q[5],q[43];
cx q[6],q[62];
s q[43];
cx q[29],q[60];
s q[51];
s q[28];
cx q[52],q[5];
s q[9];
s q[30];
h q[0];
s q[53];
h q[0];
h q[35];
h q[48];
s q[58];
h q[2];
cx q[26],q[60];
h q[16];
s q[62];
s q[56];
h q[15];
s q[39];
h q[25];
h q[64];
h q[17];
cx q[47],q[53];
cx q[24],q[35];
cx q[50],q[39];
cx q[62],q[28];
h q[59];
cx q[1],q[15];
h q[18];
h q[17];
s q[59];
s q[5];
h q[10];
h q[29];
s q[55];
h q[64];
s q[41];
s q[21];
cx q[39],q[2];
cx q[49],q[56];
cx q[27],q[9];
cx q[51],q[48];
h q[8];
s q[48];
h q[59];
cx q[14],q[22];
cx q[47],q[52];
h q[59];
h q[57];
cx q[4],q[58];
h q[22];
cx q[46],q[44];
s q[52];
s q[1];
s q[38];
cx q[52],q[0];
s q[7];
s q[49];
s q[28];
