OPENQASM 2.0;
include "qelib1.inc";
qreg q[65];
s q[42];
cx q[1],q[6];
cx q[14],q[47];
cx q[35],q[42];
cx q[33],q[21];
s q[46];
cx q[28],q[29];
h q[19];
cx q[62],q[61];
cx q[31],q[0];
s q[18];
h q[5];
s q[42];
cx q[25],q[63];
h q[15];
cx q[4],q[21];
s q[31];
h q[14];
s q[60];
h q[11];
s q[42];
cx q[41],q[48];
h q[18];
s q[31];
h q[21];
h q[64];
s q[13];
h q[51];
cx q[10],q[55];
cx q[9],q[35];
s q[52];
cx q[37],q[63];
cx q[20],q[38];
s q[52];
s q[29];
cx q[0],q[5];
s q[21];
h q[24];
h q[12];
h q[64];
s q[3];
s q[6];
s q[53];
cx q[17],q[24];
s q[13];
cx q[17],q[48];
h q[21];
cx q[42],q[58];
cx q[24],q[53];
s q[47];
cx q[37],q[48];
s q[56];
h q[19];
s q[10];
cx q[63],q[27];
cx q[39],q[38];
h q[14];
s q[24];
cx q[41],q[35];
cx q[27],q[14];
h q[56];
h q[64];
h q[32];
h q[55];
cx q[45],q[58];
cx q[17],q[7];
h q[54];
s q[14];
h q[34];
h q[47];
s q[26];
h q[27];
s q[6];
h q[47];
h q[54];
cx q[38],q[60];
s q[24];
s q[46];
h q[16];
s q[56];
cx q[28],q[13];
cx q[11],q[23];
h q[49];
s q[28];
s q[36];
h q[50];
s q[13];
h q[6];
h q[24];
cx q[28],q[45];
s q[24];
s q[2];
h q[46];
h q[63];
s q[31];
s q[16];
h q[37];
s q[45];
cx q[47],q[17];
h q[47];
cx q[11],q[45];
cx q[39],q[4];
h q[43];
s q[0];
h q[38];
s q[32];
cx q[8],q[12];
cx q[35],q[9];
cx q[50],q[10];
cx q[53],q[17];
h q[18];
h q[46];
s q[45];
s q[30];
cx q[51],q[37];
s q[62];
s q[36];
h q[49];
s q[32];
cx q[35],q[11];
h q[4];
h q[19];
cx q[25],q[12];
s q[23];
h q[18];
s q[5];
cx q[52],q[4];
h q[28];
h q[42];
h q[27];
h q[15];
cx q[61],q[45];
cx q[3],q[33];
s q[34];
h q[62];
s q[21];
h q[12];
s q[23];
s q[32];
cx q[38],q[17];
cx q[31],q[29];
h q[2];
s q[28];
cx q[47],q[58];
h q[48];
h q[46];
h q[24];
s q[18];
s q[31];
s q[59];
cx q[31],q[50];
cx q[53],q[30];
cx q[23],q[14];
s q[54];
cx q[47],q[52];
s q[7];
h q[9];
cx q[37],q[3];
h q[36];
h q[9];
s q[11];
cx q[27],q[3];
h q[28];
cx q[7],q[9];
cx q[29],q[37];
h q[61];
cx q[40],q[32];
s q[3];
h q[21];
cx q[48],q[22];
h q[32];
s q[33];
cx q[58],q[46];
cx q[51],q[64];
h q[56];
s q[55];
cx q[30],q[52];
h q[35];
s q[24];
cx q[53],q[5];
cx q[40],q[24];
cx q[30],q[20];
s q[3];
h q[44];
s q[60];
cx q[41],q[1];
s q[63];
s q[22];
s q[36];
s q[55];
h q[12];
s q[48];
cx q[58],q[51];
cx q[10],q[30];
cx q[34],q[56];
cx q[27],q[38];
s q[25];
cx q[40],q[59];
cx q[49],q[62];
s q[50];
cx q[2],q[1];
cx q[32],q[61];
s q[8];
cx q[52],q[19];
s q[13];
s q[3];
cx q[51],q[41];
h q[60];
s q[46];
h q[30];
h q[17];
cx q[16],q[0];
s q[43];
h q[24];
h q[46];
s q[40];
s q[13];
h q[3];
h q[17];
h q[40];
s q[63];
s q[51];
h q[62];
cx q[45],q[39];
cx q[48],q[13];
cx q[27],q[53];
cx q[40],q[39];
cx q[4],q[11];
cx q[7],q[41];
h q[4];
cx q[45],q[61];
h q[12];
cx q[28],q[9];
h q[8];
s q[17];
cx q[20],q[58];
h q[48];
cx q[1],q[37];
s q[24];
h q[63];
s q[41];
cx q[39],q[59];
s q[45];
h q[58];
h q[1];
cx q[22],q[25];
s q[0];
s q[4];
cx q[58],q[54];
h q[11];
cx q[50],q[29];
h q[21];
cx q[47],q[27];
cx q[22],q[49];
h q[42];
cx q[26],q[40];
s q[11];
cx q[61],q[33];
cx q[27],q[28];
s q[4];
s q[14];
h q[35];
h q[19];
cx q[33],q[54];
cx q[12],q[25];
s q[23];
h q[33];
h q[46];
cx q[47],q[59];
cx q[27],q[11];
s q[2];
h q[44];
s q[62];
s q[39];
h q[39];
cx q[4],q[34];
s q[34];
h q[63];
cx q[52],q[18];
cx q[10],q[37];
h q[28];
h q[41];
cx q[7],q[10];
cx q[26],q[1];
h q[6];
h q[4];
s q[9];
s q[18];
cx q[52],q[17];
cx q[57],q[31];
s q[12];
h q[8];
cx q[44],q[45];
s q[5];
cx q[0],q[35];
h q[1];
h q[60];
h q[61];
h q[36];
cx q[4],q[41];
cx q[58],q[44];
s q[54];
h q[8];
h q[41];
h q[64];
s q[30];
s q[8];
h q[3];
h q[44];
cx q[13],q[14];
cx q[38],q[51];
cx q[31],q[54];
h q[9];
h q[28];
h q[9];
s q[31];
s q[10];
s q[14];
cx q[28],q[63];
s q[45];
s q[34];
s q[3];
h q[57];
s q[15];
s q[8];
s q[46];
cx q[46],q[35];
cx q[19],q[22];
cx q[37],q[6];
cx q[54],q[26];
cx q[22],q[51];
h q[34];
h q[6];
cx q[23],q[63];
cx q[39],q[2];
h q[38];
s q[35];
s q[1];
h q[23];
s q[7];
h q[29];
s q[38];
cx q[57],q[56];
h q[8];
cx q[49],q[16];
cx q[58],q[38];
s q[18];
cx q[61],q[30];
cx q[16],q[48];
h q[8];
s q[19];
cx q[16],q[41];
h q[8];
s q[53];
h q[30];
s q[16];
h q[22];
s q[64];
h q[53];
h q[19];
cx q[23],q[30];
s q[34];
h q[26];
s q[6];
cx q[3],q[32];
h q[7];
cx q[26],q[30];
h q[13];
h q[34];
s q[22];
h q[24];
cx q[4],q[63];
h q[52];
cx q[41],q[40];
s q[7];
h q[14];
cx q[10],q[29];
h q[48];
h q[39];
s q[33];
h q[56];
cx q[60],q[22];
cx q[22],q[27];
cx q[62],q[44];
h q[57];
h q[23];
cx q[35],q[23];
cx q[21],q[29];
h q[18];
s q[31];
s q[28];
h q[32];
h q[0];
h q[12];
s q[21];
h q[46];
s q[44];
h q[64];
h q[10];
h q[57];
h q[19];
cx q[39],q[7];
s q[42];
cx q[48],q[47];
s q[60];
h q[11];
s q[61];
cx q[18],q[36];
cx q[59],q[20];
h q[41];
cx q[13],q[15];
cx q[15],q[9];
h q[24];
h q[23];
s q[49];
h q[36];
cx q[29],q[52];
cx q[11],q[45];
s q[2];
cx q[7],q[14];
s q[26];
s q[54];
cx q[11],q[39];
cx q[33],q[18];
s q[55];
h q[53];
s q[0];
h q[62];
s q[37];
h q[59];
cx q[12],q[35];
s q[16];
h q[23];
s q[28];
cx q[55],q[37];
s q[64];
cx q[44],q[30];
h q[12];
cx q[39],q[47];
s q[37];
h q[17];
h q[1];
cx q[29],q[26];
h q[38];
cx q[13],q[48];
cx q[59],q[27];
h q[49];
cx q[18],q[41];
s q[19];
h q[21];
s q[8];
cx q[5],q[49];
h q[0];
cx q[20],q[7];
s q[53];
s q[6];
s q[20];
s q[48];
cx q[26],q[19];
h q[41];
cx q[37],q[41];
h q[12];
s q[36];
s q[11];
h q[5];
cx q[64],q[49];
h q[26];
s q[50];
h q[58];
h q[8];
cx q[32],q[9];
h q[48];
cx q[22],q[32];
cx q[3],q[29];
cx q[34],q[21];
cx q[18],q[24];
cx q[23],q[16];
s q[53];
h q[45];
h q[43];
cx q[54],q[13];
cx q[30],q[49];
s q[49];
s q[64];
s q[8];
cx q[22],q[3];
h q[4];
s q[17];
s q[53];
h q[40];
cx q[58],q[59];
s q[48];
cx q[9],q[35];
cx q[12],q[42];
cx q[35],q[43];
h q[60];
h q[36];
cx q[5],q[28];
s q[8];
s q[4];
h q[1];
cx q[9],q[59];
h q[53];
s q[31];
h q[64];
s q[19];
s q[8];
cx q[13],q[52];
h q[44];
cx q[39],q[23];
s q[2];
s q[12];
s q[1];
s q[44];
h q[43];
s q[42];
h q[16];
h q[55];
s q[13];
cx q[17],q[29];
cx q[49],q[0];
h q[16];
h q[24];
s q[50];
h q[12];
cx q[6],q[1];
s q[54];
h q[28];
h q[31];
h q[56];
cx q[52],q[43];
s q[63];
cx q[9],q[13];
cx q[4],q[62];
cx q[54],q[26];
h q[13];
s q[14];
cx q[51],q[46];
h q[53];
s q[44];
cx q[40],q[35];
h q[28];
h q[61];
s q[61];
s q[52];
s q[52];
s q[35];
h q[0];
s q[49];
s q[38];
s q[48];
cx q[38],q[22];
s q[63];
h q[49];
cx q[49],q[11];
cx q[32],q[58];
cx q[39],q[1];
cx q[62],q[64];
h q[16];
s q[64];
cx q[1],q[21];
s q[46];
s q[59];
h q[35];
h q[56];
cx q[24],q[7];
cx q[24],q[36];
h q[38];
h q[60];
h q[60];
cx q[33],q[37];
s q[21];
cx q[49],q[36];
s q[54];
s q[20];
h q[12];
cx q[41],q[42];
h q[36];
s q[0];
cx q[58],q[12];
s q[24];
cx q[30],q[19];
h q[52];
cx q[43],q[52];
s q[27];
s q[16];
s q[62];
cx q[25],q[61];
cx q[10],q[35];
s q[24];
s q[34];
cx q[29],q[2];
h q[6];
s q[21];
h q[54];
cx q[55],q[60];
s q[53];
cx q[46],q[31];
s q[36];
h q[56];
h q[25];
s q[26];
s q[40];
cx q[43],q[14];
h q[60];
s q[14];
s q[52];
cx q[35],q[15];
h q[41];
cx q[27],q[34];
h q[1];
s q[20];
s q[58];
s q[63];
h q[41];
h q[16];
cx q[42],q[13];
cx q[33],q[50];
s q[5];
s q[40];
cx q[2],q[16];
s q[29];
h q[22];
cx q[18],q[32];
h q[8];
cx q[8],q[32];
h q[13];
cx q[29],q[15];
s q[30];
cx q[9],q[20];
s q[55];
s q[10];
cx q[47],q[22];
s q[36];
cx q[11],q[10];
h q[19];
s q[27];
cx q[26],q[0];
s q[7];
cx q[60],q[44];
cx q[59],q[45];
cx q[51],q[33];
h q[12];
h q[20];
h q[9];
h q[57];
h q[28];
cx q[15],q[43];
h q[10];
h q[51];
h q[49];
h q[11];
cx q[45],q[9];
cx q[51],q[12];
s q[49];
cx q[43],q[59];
h q[4];
cx q[44],q[34];
cx q[24],q[20];
h q[6];
h q[11];
s q[27];
s q[47];
h q[44];
h q[49];
cx q[29],q[12];
cx q[62],q[12];
s q[52];
cx q[41],q[26];
h q[30];
h q[13];
h q[51];
s q[49];
h q[26];
cx q[55],q[42];
h q[16];
cx q[7],q[13];
s q[29];
h q[42];
s q[13];
h q[8];
s q[17];
h q[1];
cx q[26],q[5];
cx q[25],q[50];
cx q[1],q[52];
h q[37];
h q[56];
h q[8];
s q[18];
s q[57];
h q[41];
cx q[39],q[64];
s q[54];
h q[23];
s q[38];
cx q[26],q[54];
h q[47];
h q[34];
h q[31];
cx q[3],q[16];
h q[57];
s q[39];
s q[64];
s q[38];
s q[13];
s q[41];
s q[0];
s q[9];
s q[4];
s q[21];
h q[6];
cx q[33],q[57];
cx q[42],q[53];
cx q[51],q[45];
cx q[48],q[41];
cx q[29],q[34];
h q[34];
s q[12];
s q[31];
h q[28];
cx q[6],q[30];
s q[21];
h q[28];
s q[3];
h q[46];
s q[22];
cx q[24],q[54];
cx q[2],q[7];
h q[7];
h q[54];
s q[41];
cx q[7],q[54];
cx q[10],q[11];
cx q[52],q[42];
h q[47];
s q[33];
h q[47];
cx q[28],q[16];
s q[61];
h q[24];
cx q[62],q[14];
h q[34];
s q[54];
h q[28];
s q[19];
h q[63];
cx q[42],q[2];
s q[58];
cx q[27],q[23];
cx q[45],q[59];
h q[5];
s q[58];
h q[37];
h q[30];
cx q[45],q[4];
cx q[37],q[52];
h q[53];
h q[4];
cx q[44],q[31];
cx q[30],q[32];
cx q[47],q[38];
h q[52];
h q[48];
h q[48];
h q[23];
h q[62];
h q[61];
h q[56];
h q[31];
h q[55];
h q[10];
h q[11];
s q[19];
cx q[38],q[25];
h q[58];
cx q[64],q[40];
s q[3];
h q[24];
cx q[14],q[50];
h q[64];
h q[46];
cx q[14],q[29];
h q[58];
cx q[25],q[3];
cx q[51],q[57];
h q[29];
s q[5];
h q[34];
s q[26];
h q[52];
h q[63];
s q[39];
cx q[44],q[20];
cx q[35],q[2];
cx q[23],q[40];
s q[25];
s q[34];
s q[10];
h q[17];
cx q[12],q[10];
s q[21];
cx q[11],q[37];
h q[4];
cx q[13],q[23];
cx q[51],q[25];
s q[58];
h q[42];
s q[18];
h q[6];
h q[45];
cx q[58],q[37];
h q[49];
cx q[18],q[2];
h q[64];
s q[64];
s q[21];
h q[54];
cx q[30],q[23];
s q[19];
cx q[41],q[7];
cx q[15],q[16];
cx q[7],q[13];
h q[33];
h q[34];
cx q[47],q[46];
h q[5];
h q[63];
cx q[49],q[20];
h q[24];
s q[62];
cx q[47],q[39];
cx q[20],q[0];
s q[39];
cx q[28],q[8];
s q[50];
s q[52];
cx q[51],q[40];
s q[31];
h q[28];
s q[20];
s q[62];
cx q[11],q[24];
s q[36];
cx q[4],q[58];
cx q[38],q[41];
h q[42];
cx q[57],q[58];
h q[22];
s q[2];
cx q[0],q[2];
s q[4];
cx q[60],q[12];
s q[29];
cx q[52],q[3];
s q[48];
h q[61];
cx q[22],q[24];
cx q[25],q[4];
s q[44];
s q[55];
cx q[56],q[17];
h q[35];
h q[44];
cx q[63],q[34];
s q[26];
s q[62];
h q[56];
h q[64];
h q[6];
h q[53];
h q[63];
s q[19];
s q[41];
cx q[17],q[20];
s q[62];
cx q[39],q[56];
s q[31];
h q[10];
s q[13];
s q[57];
s q[28];
cx q[18],q[48];
s q[53];
cx q[3],q[28];
cx q[8],q[41];
cx q[26],q[5];
h q[26];
cx q[60],q[7];
h q[6];
s q[41];
s q[41];
cx q[44],q[14];
s q[8];
cx q[12],q[15];
cx q[34],q[61];
cx q[23],q[35];
s q[33];
s q[60];
s q[50];
s q[35];
s q[8];
h q[62];
s q[26];
s q[17];
s q[10];
cx q[24],q[31];
s q[46];
h q[30];
s q[59];
cx q[56],q[4];
h q[39];
cx q[24],q[17];
cx q[23],q[26];
cx q[42],q[41];
h q[19];
h q[55];
s q[5];
h q[3];
h q[13];
h q[36];
cx q[2],q[6];
cx q[49],q[1];
cx q[28],q[30];
cx q[29],q[23];
cx q[36],q[24];
cx q[21],q[53];
cx q[20],q[24];
cx q[52],q[46];
h q[34];
s q[40];
s q[24];
cx q[32],q[5];
h q[44];
cx q[22],q[46];
cx q[5],q[22];
h q[40];
s q[59];
cx q[39],q[59];
s q[60];
h q[26];
s q[54];
s q[13];
cx q[27],q[34];
s q[29];
h q[39];
s q[57];
cx q[21],q[18];
h q[8];
h q[61];
h q[18];
cx q[9],q[25];
s q[36];
cx q[53],q[24];
s q[24];
s q[21];
s q[32];
s q[53];
cx q[15],q[25];
s q[16];
h q[27];
cx q[13],q[42];
cx q[30],q[48];
cx q[52],q[44];
s q[44];
cx q[48],q[39];
s q[25];
h q[46];
s q[25];
cx q[50],q[5];
s q[20];
h q[28];
cx q[51],q[22];
s q[51];
cx q[3],q[40];
s q[23];
s q[22];
h q[63];
s q[38];
s q[11];
cx q[23],q[47];
s q[3];
h q[30];
cx q[40],q[26];
s q[56];
cx q[57],q[43];
cx q[17],q[24];
h q[34];
cx q[21],q[38];
s q[51];
cx q[2],q[5];
s q[22];
h q[2];
h q[7];
h q[61];
s q[53];
cx q[34],q[47];
h q[42];
h q[48];
s q[15];
cx q[57],q[15];
h q[12];
cx q[19],q[22];
h q[60];
h q[16];
cx q[2],q[13];
cx q[33],q[16];
h q[39];
cx q[49],q[34];
s q[52];
h q[61];
h q[6];
cx q[24],q[59];
s q[51];
cx q[35],q[37];
s q[11];
s q[20];
s q[7];
s q[0];
cx q[11],q[59];
s q[24];
cx q[14],q[25];
cx q[49],q[0];
s q[59];
h q[23];
s q[25];
cx q[23],q[17];
s q[10];
cx q[12],q[28];
s q[9];
h q[28];
s q[4];
h q[26];
cx q[30],q[26];
cx q[38],q[5];
h q[14];
s q[7];
h q[57];
h q[60];
s q[60];
s q[60];
s q[49];
h q[28];
s q[46];
h q[0];
s q[27];
h q[64];
s q[48];
s q[63];
s q[42];
cx q[30],q[44];
cx q[10],q[61];
s q[37];
s q[53];
s q[11];
s q[26];
h q[2];
s q[11];
h q[42];
s q[2];
s q[57];
s q[55];
h q[22];
h q[35];
cx q[8],q[21];
h q[51];
s q[4];
cx q[17],q[56];
s q[32];
h q[32];
cx q[14],q[28];
h q[4];
s q[0];
s q[37];
h q[51];
s q[24];
h q[16];
h q[62];
s q[42];
cx q[46],q[25];
s q[5];
s q[19];
h q[22];
s q[60];
h q[45];
s q[23];
cx q[64],q[12];
cx q[1],q[37];
cx q[35],q[0];
h q[7];
s q[50];
h q[2];
s q[30];
s q[13];
cx q[48],q[24];
cx q[52],q[29];
s q[28];
s q[52];
h q[1];
h q[28];
h q[9];
cx q[23],q[46];
cx q[49],q[46];
s q[5];
cx q[23],q[35];
cx q[28],q[37];
cx q[11],q[53];
h q[57];
h q[38];
s q[35];
cx q[28],q[64];
h q[1];
cx q[16],q[35];
h q[21];
h q[51];
h q[34];
cx q[59],q[46];
s q[44];
cx q[62],q[44];
s q[45];
h q[34];
s q[49];
h q[4];
s q[58];
h q[59];
cx q[36],q[56];
h q[32];
s q[1];
cx q[30],q[51];
cx q[21],q[48];
cx q[56],q[44];
cx q[2],q[20];
cx q[40],q[6];
h q[24];
s q[14];
cx q[21],q[44];
cx q[54],q[27];
h q[0];
s q[10];
s q[20];
h q[22];
h q[35];
s q[20];
h q[3];
s q[4];
cx q[2],q[31];
cx q[43],q[50];
cx q[20],q[39];
cx q[46],q[53];
s q[12];
h q[51];
s q[19];
s q[38];
h q[23];
cx q[41],q[34];
h q[58];
cx q[11],q[23];
h q[57];
cx q[40],q[49];
s q[40];
h q[5];
cx q[27],q[61];
h q[30];
cx q[50],q[40];
h q[59];
s q[48];
h q[16];
s q[6];
h q[26];
h q[37];
cx q[49],q[50];
cx q[59],q[62];
cx q[52],q[4];
h q[64];
cx q[26],q[24];
s q[9];
h q[39];
h q[25];
h q[45];
h q[59];
s q[41];
cx q[53],q[44];
s q[15];
h q[27];
cx q[45],q[50];
h q[39];
s q[35];
cx q[30],q[20];
cx q[17],q[36];
h q[6];
cx q[27],q[18];
cx q[63],q[3];
h q[47];
h q[5];
cx q[39],q[49];
cx q[28],q[9];
h q[59];
s q[48];
h q[19];
h q[24];
cx q[23],q[61];
h q[51];
s q[4];
h q[34];
cx q[39],q[38];
s q[43];
cx q[26],q[37];
cx q[63],q[10];
cx q[13],q[10];
s q[1];
cx q[56],q[20];
s q[12];
s q[62];
h q[12];
h q[16];
h q[11];
s q[46];
s q[47];
cx q[37],q[1];
cx q[2],q[45];
s q[14];
h q[9];
h q[0];
cx q[27],q[64];
h q[55];
h q[52];
cx q[39],q[21];
cx q[23],q[28];
h q[16];
s q[33];
cx q[22],q[23];
cx q[62],q[29];
cx q[24],q[37];
s q[6];
s q[13];
h q[11];
h q[16];
h q[14];
h q[40];
h q[30];
h q[48];
s q[50];
s q[49];
h q[22];
cx q[3],q[50];
h q[38];
cx q[36],q[49];
s q[20];
s q[60];
s q[47];
s q[4];
s q[28];
cx q[4],q[52];
cx q[10],q[55];
cx q[50],q[24];
h q[27];
cx q[45],q[37];
h q[14];
s q[54];
s q[25];
s q[23];
cx q[7],q[32];
h q[19];
s q[48];
h q[41];
cx q[8],q[61];
s q[63];
cx q[35],q[22];
cx q[25],q[2];
s q[22];
cx q[46],q[25];
h q[25];
s q[17];
s q[55];
s q[58];
s q[3];
h q[54];
s q[12];
cx q[34],q[35];
h q[26];
cx q[62],q[29];
cx q[27],q[47];
cx q[10],q[33];
h q[22];
s q[42];
s q[14];
s q[64];
h q[28];
cx q[49],q[40];
cx q[42],q[34];
s q[6];
h q[6];
h q[26];
h q[50];
h q[55];
cx q[60],q[55];
h q[34];
cx q[34],q[56];
s q[1];
s q[64];
h q[56];
s q[1];
cx q[64],q[31];
h q[39];
s q[40];
h q[31];
cx q[35],q[18];
cx q[3],q[33];
h q[46];
s q[19];
s q[51];
h q[5];
h q[21];
s q[48];
s q[15];
cx q[46],q[22];
s q[52];
s q[15];
h q[55];
s q[38];
h q[46];
s q[24];
s q[17];
h q[54];
s q[42];
s q[32];
h q[18];
h q[51];
cx q[4],q[24];
s q[25];
s q[5];
s q[36];
h q[2];
h q[24];
s q[52];
h q[61];
h q[36];
h q[26];
cx q[55],q[41];
h q[15];
cx q[21],q[59];
s q[51];
h q[13];
s q[10];
cx q[64],q[30];
s q[37];
cx q[55],q[26];
s q[28];
s q[11];
s q[25];
s q[10];
h q[39];
h q[35];
h q[50];
h q[50];
cx q[59],q[14];
s q[23];
s q[28];
s q[36];
cx q[5],q[50];
cx q[5],q[47];
h q[63];
h q[50];
h q[28];
cx q[28],q[10];
h q[39];
cx q[59],q[54];
cx q[61],q[6];
cx q[15],q[50];
s q[26];
h q[64];
h q[42];
s q[19];
cx q[2],q[48];
h q[28];
cx q[19],q[44];
s q[0];
s q[43];
cx q[28],q[16];
s q[16];
h q[12];
cx q[10],q[21];
cx q[41],q[57];
s q[32];
s q[23];
s q[18];
h q[34];
s q[46];
cx q[41],q[61];
cx q[45],q[15];
cx q[4],q[21];
s q[28];
h q[58];
h q[17];
s q[58];
cx q[11],q[23];
h q[48];
s q[46];
s q[27];
h q[54];
s q[37];
s q[37];
s q[21];
cx q[44],q[62];
h q[55];
h q[20];
s q[45];
s q[50];
cx q[3],q[12];
s q[50];
s q[39];
s q[60];
cx q[2],q[12];
s q[51];
s q[3];
h q[29];
s q[49];
s q[47];
h q[22];
cx q[20],q[14];
h q[45];
h q[43];
cx q[18],q[19];
h q[7];
cx q[23],q[5];
cx q[15],q[41];
cx q[11],q[54];
h q[26];
cx q[19],q[44];
cx q[64],q[59];
h q[22];
h q[1];
s q[35];
cx q[16],q[28];
cx q[13],q[2];
s q[15];
s q[55];
s q[46];
h q[40];
cx q[45],q[5];
cx q[39],q[10];
h q[35];
cx q[20],q[25];
h q[55];
h q[20];
h q[58];
s q[19];
cx q[63],q[30];
s q[55];
h q[63];
s q[53];
cx q[54],q[17];
h q[50];
cx q[46],q[18];
cx q[2],q[62];
s q[5];
cx q[51],q[3];
h q[22];
h q[25];
h q[64];
h q[30];
h q[7];
s q[49];
h q[62];
s q[26];
h q[8];
h q[44];
h q[26];
s q[56];
s q[2];
cx q[6],q[50];
cx q[29],q[59];
cx q[25],q[36];
s q[52];
s q[3];
s q[60];
h q[21];
s q[57];
h q[4];
s q[54];
s q[59];
cx q[43],q[29];
h q[37];
cx q[47],q[58];
h q[22];
h q[40];
s q[32];
s q[37];
s q[24];
cx q[48],q[30];
s q[4];
cx q[5],q[54];
cx q[51],q[15];
h q[43];
cx q[24],q[22];
cx q[9],q[56];
s q[7];
cx q[22],q[17];
cx q[6],q[62];
s q[13];
s q[44];
s q[49];
cx q[29],q[52];
cx q[42],q[22];
s q[33];
h q[32];
h q[62];
cx q[22],q[49];
cx q[46],q[13];
s q[13];
s q[38];
s q[10];
cx q[2],q[10];
s q[62];
cx q[17],q[61];
s q[43];
s q[26];
cx q[50],q[29];
s q[31];
h q[42];
cx q[53],q[5];
s q[43];
h q[29];
cx q[53],q[58];
h q[10];
s q[31];
s q[64];
cx q[38],q[55];
h q[50];
s q[22];
h q[11];
cx q[30],q[19];
h q[6];
cx q[38],q[20];
cx q[19],q[25];
h q[10];
cx q[25],q[55];
s q[62];
cx q[4],q[32];
h q[48];
h q[41];
h q[46];
cx q[60],q[16];
cx q[14],q[11];
h q[31];
h q[35];
h q[6];
s q[20];
h q[14];
s q[64];
h q[47];
cx q[47],q[21];
s q[0];
s q[49];
cx q[51],q[34];
s q[3];
s q[49];
h q[14];
h q[64];
cx q[59],q[33];
cx q[9],q[38];
h q[34];
cx q[13],q[56];
s q[16];
cx q[17],q[47];
h q[33];
s q[46];
h q[64];
cx q[29],q[0];
s q[60];
s q[39];
s q[58];
h q[55];
s q[22];
cx q[14],q[11];
s q[51];
s q[15];
cx q[40],q[0];
cx q[54],q[10];
cx q[42],q[48];
cx q[15],q[25];
s q[0];
s q[42];
s q[64];
h q[7];
h q[56];
s q[54];
cx q[51],q[35];
h q[60];
s q[10];
cx q[53],q[16];
cx q[42],q[56];
cx q[8],q[54];
cx q[49],q[52];
cx q[0],q[5];
h q[44];
h q[2];
s q[49];
cx q[64],q[25];
s q[37];
cx q[24],q[42];
h q[23];
h q[37];
s q[3];
h q[62];
s q[53];
cx q[47],q[54];
s q[34];
s q[43];
h q[35];
s q[15];
cx q[40],q[8];
h q[19];
s q[50];
h q[50];
s q[46];
cx q[10],q[25];
h q[48];
s q[29];
cx q[38],q[48];
cx q[10],q[52];
h q[24];
h q[37];
cx q[8],q[42];
h q[41];
h q[59];
cx q[44],q[54];
h q[48];
s q[40];
cx q[55],q[48];
cx q[18],q[55];
s q[12];
s q[13];
s q[25];
s q[4];
cx q[51],q[40];
s q[2];
h q[29];
s q[18];
h q[35];
h q[54];
h q[31];
cx q[52],q[8];
s q[25];
h q[51];
s q[40];
cx q[18],q[24];
s q[17];
cx q[56],q[39];
cx q[10],q[19];
cx q[43],q[25];
s q[38];
h q[4];
cx q[55],q[37];
cx q[62],q[12];
cx q[34],q[58];
s q[14];
h q[44];
cx q[32],q[24];
s q[48];
s q[0];
cx q[26],q[5];
h q[33];
s q[10];
h q[6];
s q[62];
h q[23];
s q[60];
cx q[50],q[5];
cx q[52],q[53];
h q[4];
s q[52];
h q[25];
h q[43];
h q[25];
cx q[8],q[43];
s q[49];
h q[16];
cx q[44],q[46];
h q[43];
h q[48];
h q[48];
cx q[57],q[6];
s q[63];
h q[62];
cx q[61],q[2];
h q[24];
h q[4];
h q[15];
s q[0];
s q[24];
h q[44];
s q[49];
s q[46];
cx q[45],q[35];
cx q[23],q[33];
h q[12];
h q[57];
cx q[32],q[42];
cx q[5],q[43];
s q[41];
cx q[18],q[11];
h q[25];
h q[25];
cx q[8],q[32];
cx q[56],q[25];
s q[63];
cx q[23],q[48];
h q[29];
s q[53];
s q[11];
cx q[35],q[26];
s q[32];
cx q[17],q[1];
s q[56];
h q[57];
s q[41];
s q[42];
s q[15];
s q[49];
h q[15];
s q[44];
s q[44];
s q[20];
cx q[49],q[20];
s q[24];
h q[28];
cx q[10],q[2];
h q[32];
cx q[14],q[29];
cx q[11],q[60];
cx q[28],q[2];
cx q[35],q[34];
cx q[26],q[30];
s q[46];
h q[29];
cx q[57],q[62];
s q[20];
h q[47];
h q[18];
h q[64];
h q[32];
s q[24];
cx q[56],q[62];
cx q[43],q[4];
cx q[27],q[63];
cx q[2],q[53];
cx q[29],q[14];
s q[48];
s q[43];
h q[60];
cx q[55],q[39];
h q[50];
cx q[45],q[40];
h q[9];
cx q[37],q[38];
h q[34];
h q[63];
h q[16];
cx q[2],q[48];
h q[45];
h q[26];
h q[36];
cx q[49],q[30];
h q[26];
cx q[53],q[35];
h q[50];
h q[8];
h q[51];
cx q[34],q[47];
h q[22];
h q[61];
cx q[29],q[19];
s q[15];
cx q[14],q[43];
cx q[56],q[57];
h q[22];
cx q[50],q[29];
cx q[17],q[61];
cx q[23],q[3];
s q[6];
h q[40];
cx q[61],q[16];
h q[12];
cx q[28],q[24];
s q[47];
s q[37];
h q[22];
h q[15];
h q[58];
cx q[30],q[25];
cx q[19],q[46];
h q[5];
cx q[19],q[18];
s q[36];
h q[53];
h q[16];
s q[56];
h q[60];
s q[4];
h q[47];
