OPENQASM 2.0;
include "qelib1.inc";
qreg q[65];
cx q[63],q[33];
h q[40];
s q[59];
s q[4];
cx q[39],q[46];
cx q[38],q[29];
cx q[13],q[36];
h q[14];
h q[25];
h q[35];
s q[38];
h q[34];
cx q[17],q[56];
s q[20];
s q[23];
cx q[31],q[34];
cx q[24],q[38];
s q[45];
h q[30];
h q[31];
s q[2];
s q[62];
cx q[24],q[49];
cx q[23],q[36];
h q[2];
h q[17];
s q[27];
s q[60];
s q[46];
cx q[20],q[13];
s q[54];
s q[45];
s q[29];
cx q[21],q[36];
h q[52];
s q[19];
h q[34];
cx q[11],q[10];
cx q[5],q[64];
s q[1];
s q[15];
h q[17];
cx q[50],q[6];
h q[35];
s q[11];
h q[35];
s q[51];
s q[32];
h q[26];
cx q[58],q[15];
h q[21];
h q[46];
h q[53];
h q[58];
cx q[9],q[45];
s q[63];
h q[56];
h q[28];
h q[32];
h q[28];
cx q[35],q[22];
h q[37];
s q[33];
h q[42];
cx q[26],q[43];
h q[43];
s q[28];
s q[29];
s q[55];
cx q[7],q[34];
cx q[25],q[53];
h q[55];
s q[11];
cx q[49],q[39];
h q[42];
s q[1];
s q[30];
h q[29];
s q[32];
s q[60];
s q[26];
h q[62];
h q[56];
cx q[48],q[41];
h q[20];
cx q[14],q[20];
cx q[7],q[43];
s q[21];
s q[62];
cx q[49],q[18];
h q[25];
h q[48];
cx q[19],q[52];
s q[1];
s q[25];
h q[61];
cx q[31],q[2];
s q[64];
cx q[37],q[18];
h q[18];
s q[16];
h q[11];
cx q[5],q[11];
h q[56];
cx q[14],q[2];
h q[14];
s q[10];
cx q[23],q[17];
h q[33];
cx q[10],q[35];
s q[1];
h q[61];
s q[42];
h q[44];
s q[23];
h q[50];
h q[40];
cx q[21],q[17];
h q[15];
cx q[36],q[37];
h q[15];
s q[10];
h q[28];
h q[54];
h q[43];
h q[14];
s q[26];
h q[0];
h q[13];
h q[31];
h q[14];
cx q[40],q[35];
h q[29];
h q[0];
s q[62];
s q[36];
h q[46];
cx q[30],q[20];
cx q[14],q[61];
h q[11];
cx q[14],q[21];
s q[31];
h q[8];
h q[42];
s q[50];
s q[34];
cx q[60],q[31];
h q[42];
s q[19];
cx q[64],q[15];
s q[15];
s q[52];
cx q[13],q[25];
s q[0];
h q[19];
cx q[15],q[45];
h q[39];
h q[28];
h q[45];
s q[63];
cx q[17],q[63];
cx q[2],q[54];
s q[50];
s q[60];
s q[15];
h q[25];
h q[61];
h q[55];
cx q[5],q[23];
s q[55];
cx q[59],q[12];
h q[43];
s q[14];
h q[60];
h q[43];
s q[8];
h q[18];
h q[44];
h q[44];
h q[37];
s q[53];
s q[6];
h q[50];
s q[33];
h q[13];
s q[58];
cx q[58],q[45];
h q[13];
h q[50];
s q[38];
h q[32];
s q[30];
cx q[22],q[49];
h q[31];
cx q[64],q[62];
cx q[58],q[11];
h q[1];
cx q[26],q[12];
s q[50];
cx q[52],q[56];
s q[49];
h q[15];
cx q[26],q[35];
h q[17];
s q[49];
cx q[36],q[64];
h q[41];
s q[38];
s q[1];
h q[21];
cx q[36],q[26];
h q[45];
h q[60];
h q[13];
cx q[32],q[60];
cx q[28],q[58];
h q[37];
h q[4];
s q[0];
h q[3];
s q[5];
h q[63];
s q[27];
s q[38];
h q[17];
s q[45];
s q[21];
s q[8];
cx q[31],q[25];
s q[35];
s q[52];
cx q[36],q[63];
cx q[34],q[30];
cx q[37],q[57];
s q[45];
s q[31];
s q[13];
s q[4];
h q[0];
cx q[19],q[22];
cx q[19],q[1];
h q[6];
h q[12];
cx q[13],q[6];
s q[39];
h q[6];
s q[9];
cx q[31],q[56];
s q[19];
s q[50];
h q[42];
cx q[53],q[45];
cx q[0],q[17];
h q[23];
cx q[51],q[9];
s q[47];
cx q[54],q[10];
cx q[2],q[5];
cx q[1],q[3];
s q[44];
h q[51];
s q[32];
cx q[46],q[31];
cx q[10],q[61];
s q[62];
h q[16];
cx q[12],q[48];
h q[31];
h q[52];
h q[44];
cx q[57],q[53];
cx q[37],q[21];
cx q[19],q[4];
cx q[2],q[57];
s q[12];
cx q[3],q[4];
h q[51];
h q[20];
h q[21];
cx q[6],q[39];
cx q[62],q[34];
h q[1];
h q[13];
cx q[13],q[52];
cx q[13],q[51];
cx q[7],q[46];
h q[9];
h q[20];
s q[40];
cx q[18],q[41];
cx q[6],q[18];
s q[12];
h q[0];
h q[62];
h q[15];
s q[27];
cx q[64],q[14];
s q[43];
h q[39];
cx q[49],q[44];
s q[36];
h q[47];
s q[15];
s q[8];
cx q[27],q[14];
cx q[29],q[53];
cx q[40],q[52];
cx q[57],q[43];
s q[2];
s q[33];
h q[42];
s q[10];
s q[3];
cx q[4],q[21];
cx q[41],q[18];
h q[53];
cx q[26],q[40];
h q[3];
h q[15];
h q[22];
cx q[4],q[48];
cx q[26],q[48];
h q[28];
h q[35];
s q[37];
s q[5];
h q[10];
h q[53];
h q[1];
cx q[32],q[34];
h q[62];
h q[12];
h q[26];
cx q[38],q[15];
h q[41];
cx q[51],q[6];
h q[2];
cx q[6],q[4];
s q[53];
h q[6];
s q[29];
cx q[60],q[7];
s q[55];
cx q[18],q[54];
s q[22];
cx q[7],q[54];
h q[42];
s q[42];
h q[30];
s q[18];
s q[38];
h q[37];
s q[53];
cx q[8],q[59];
s q[40];
s q[46];
h q[41];
cx q[31],q[43];
s q[47];
s q[15];
h q[8];
cx q[49],q[25];
s q[38];
s q[11];
s q[38];
cx q[59],q[44];
h q[5];
cx q[31],q[50];
cx q[10],q[30];
h q[40];
h q[12];
h q[64];
h q[47];
cx q[26],q[11];
cx q[11],q[22];
cx q[23],q[22];
h q[27];
s q[14];
h q[1];
s q[23];
h q[23];
h q[34];
h q[44];
h q[23];
h q[63];
s q[47];
cx q[22],q[1];
cx q[25],q[31];
h q[26];
s q[23];
h q[61];
s q[37];
h q[1];
cx q[1],q[57];
cx q[40],q[9];
s q[38];
cx q[21],q[51];
s q[12];
s q[26];
s q[28];
cx q[59],q[49];
h q[53];
h q[43];
s q[43];
cx q[20],q[62];
s q[58];
cx q[23],q[33];
h q[49];
cx q[34],q[26];
cx q[56],q[51];
s q[39];
cx q[60],q[22];
s q[48];
cx q[35],q[48];
h q[8];
h q[40];
s q[38];
h q[11];
s q[19];
cx q[56],q[26];
s q[39];
cx q[1],q[45];
cx q[60],q[44];
h q[32];
cx q[39],q[47];
s q[18];
s q[59];
h q[10];
h q[5];
cx q[21],q[10];
h q[46];
cx q[26],q[60];
cx q[4],q[15];
cx q[10],q[36];
h q[18];
h q[58];
h q[58];
h q[20];
cx q[49],q[5];
h q[26];
s q[16];
h q[19];
h q[2];
h q[45];
s q[51];
cx q[54],q[58];
cx q[64],q[9];
h q[58];
h q[41];
h q[19];
s q[49];
h q[45];
s q[3];
s q[28];
s q[21];
cx q[17],q[0];
s q[18];
cx q[45],q[24];
cx q[17],q[36];
h q[53];
cx q[57],q[8];
cx q[6],q[39];
h q[54];
cx q[13],q[7];
s q[30];
s q[60];
cx q[4],q[38];
h q[53];
s q[36];
h q[44];
h q[14];
s q[8];
s q[2];
cx q[32],q[44];
cx q[42],q[8];
cx q[46],q[62];
s q[60];
s q[40];
s q[13];
s q[29];
s q[62];
h q[5];
cx q[64],q[49];
s q[64];
cx q[12],q[64];
s q[32];
s q[59];
cx q[50],q[8];
cx q[18],q[56];
h q[4];
cx q[58],q[61];
h q[51];
cx q[9],q[29];
s q[13];
s q[42];
cx q[17],q[13];
h q[12];
s q[31];
s q[29];
h q[44];
h q[33];
s q[38];
s q[47];
cx q[62],q[2];
cx q[12],q[29];
s q[15];
cx q[16],q[27];
cx q[5],q[59];
h q[56];
cx q[5],q[48];
s q[22];
cx q[33],q[41];
cx q[61],q[41];
cx q[62],q[47];
cx q[42],q[5];
s q[49];
cx q[60],q[1];
h q[33];
s q[21];
cx q[34],q[57];
h q[41];
h q[15];
cx q[22],q[14];
h q[52];
s q[10];
s q[41];
cx q[25],q[39];
s q[5];
s q[57];
s q[11];
s q[60];
h q[41];
s q[16];
h q[25];
cx q[30],q[53];
cx q[60],q[55];
s q[63];
h q[0];
cx q[15],q[59];
cx q[64],q[34];
cx q[25],q[62];
cx q[58],q[63];
h q[10];
s q[34];
s q[59];
cx q[57],q[41];
s q[4];
cx q[41],q[0];
h q[23];
s q[46];
h q[26];
s q[31];
cx q[19],q[43];
s q[3];
h q[62];
s q[60];
cx q[1],q[31];
s q[12];
cx q[34],q[60];
cx q[13],q[16];
s q[1];
cx q[5],q[19];
cx q[43],q[36];
s q[15];
s q[29];
s q[29];
cx q[55],q[11];
s q[5];
cx q[28],q[52];
cx q[12],q[35];
s q[44];
h q[50];
h q[35];
h q[6];
s q[59];
s q[32];
s q[55];
s q[35];
h q[54];
s q[16];
h q[49];
h q[7];
cx q[8],q[41];
h q[26];
s q[17];
h q[35];
h q[49];
s q[59];
cx q[39],q[51];
cx q[11],q[37];
h q[36];
cx q[33],q[22];
h q[3];
cx q[51],q[1];
cx q[50],q[5];
h q[44];
cx q[39],q[47];
s q[60];
cx q[33],q[39];
s q[25];
s q[24];
cx q[11],q[36];
h q[5];
h q[27];
cx q[62],q[0];
h q[28];
h q[39];
s q[32];
cx q[9],q[39];
h q[16];
h q[1];
h q[29];
s q[60];
cx q[52],q[15];
s q[13];
h q[2];
s q[14];
cx q[35],q[33];
s q[64];
h q[43];
s q[55];
s q[18];
h q[13];
h q[29];
h q[63];
s q[16];
cx q[61],q[34];
s q[47];
s q[21];
h q[58];
h q[15];
cx q[21],q[50];
h q[18];
h q[62];
cx q[8],q[22];
s q[21];
s q[62];
h q[45];
cx q[1],q[5];
cx q[33],q[10];
cx q[35],q[10];
s q[37];
s q[55];
h q[36];
h q[22];
cx q[35],q[50];
s q[50];
h q[58];
s q[19];
h q[25];
s q[16];
h q[55];
cx q[56],q[57];
cx q[52],q[62];
s q[53];
h q[36];
s q[31];
s q[38];
cx q[25],q[55];
h q[63];
s q[53];
s q[13];
cx q[30],q[3];
cx q[45],q[1];
s q[3];
s q[50];
cx q[42],q[50];
cx q[56],q[22];
cx q[18],q[52];
cx q[50],q[25];
cx q[10],q[33];
h q[34];
cx q[56],q[34];
s q[55];
s q[12];
h q[24];
s q[26];
s q[23];
h q[28];
cx q[8],q[3];
h q[45];
s q[54];
cx q[62],q[36];
cx q[39],q[50];
s q[64];
s q[28];
cx q[19],q[22];
h q[46];
h q[47];
s q[23];
s q[6];
cx q[9],q[60];
s q[25];
h q[34];
cx q[28],q[17];
s q[58];
h q[17];
h q[12];
s q[42];
s q[4];
s q[61];
h q[63];
s q[59];
h q[6];
cx q[45],q[58];
s q[38];
s q[38];
cx q[59],q[46];
h q[32];
s q[41];
s q[32];
s q[12];
h q[31];
cx q[54],q[16];
h q[18];
h q[46];
s q[39];
h q[63];
s q[64];
s q[18];
cx q[6],q[55];
h q[43];
h q[59];
s q[36];
s q[64];
s q[7];
s q[24];
h q[58];
cx q[36],q[51];
cx q[26],q[43];
cx q[19],q[40];
h q[6];
s q[9];
s q[53];
cx q[22],q[34];
s q[26];
h q[54];
cx q[31],q[51];
s q[64];
h q[49];
cx q[16],q[18];
s q[10];
s q[56];
h q[8];
cx q[52],q[7];
s q[59];
s q[29];
cx q[46],q[42];
cx q[8],q[19];
s q[11];
cx q[0],q[5];
cx q[30],q[26];
cx q[59],q[44];
s q[47];
h q[30];
h q[54];
cx q[13],q[43];
cx q[7],q[46];
cx q[62],q[3];
h q[62];
cx q[52],q[0];
h q[62];
cx q[53],q[13];
cx q[7],q[24];
cx q[48],q[40];
cx q[38],q[57];
cx q[50],q[8];
s q[13];
cx q[8],q[55];
cx q[21],q[7];
h q[7];
cx q[24],q[54];
h q[15];
h q[39];
cx q[32],q[29];
cx q[17],q[63];
s q[40];
s q[20];
h q[19];
cx q[64],q[26];
s q[29];
cx q[12],q[2];
cx q[61],q[40];
h q[12];
cx q[60],q[32];
s q[28];
s q[36];
s q[17];
s q[18];
h q[31];
h q[56];
cx q[11],q[24];
cx q[40],q[46];
h q[35];
h q[27];
s q[14];
cx q[11],q[18];
cx q[41],q[58];
cx q[44],q[20];
cx q[46],q[27];
h q[49];
s q[8];
s q[64];
h q[21];
cx q[9],q[35];
cx q[47],q[6];
cx q[4],q[7];
cx q[7],q[31];
h q[40];
s q[44];
s q[28];
h q[55];
h q[1];
s q[33];
cx q[53],q[62];
h q[59];
cx q[1],q[61];
cx q[15],q[53];
s q[10];
cx q[22],q[36];
s q[14];
s q[60];
h q[2];
h q[45];
h q[56];
h q[15];
s q[43];
h q[50];
h q[34];
cx q[47],q[61];
h q[46];
h q[32];
cx q[36],q[34];
cx q[51],q[22];
s q[20];
s q[16];
s q[13];
h q[5];
h q[37];
h q[55];
cx q[64],q[20];
cx q[34],q[24];
h q[54];
h q[40];
s q[2];
s q[9];
h q[61];
cx q[62],q[47];
h q[40];
s q[2];
cx q[38],q[31];
s q[35];
s q[58];
s q[58];
cx q[45],q[9];
cx q[21],q[10];
s q[26];
s q[16];
cx q[40],q[38];
s q[27];
cx q[55],q[10];
h q[11];
s q[47];
h q[21];
s q[57];
h q[40];
s q[31];
cx q[33],q[29];
cx q[52],q[27];
s q[37];
s q[8];
cx q[1],q[52];
s q[49];
cx q[59],q[28];
h q[21];
s q[10];
s q[18];
h q[14];
cx q[34],q[59];
cx q[60],q[39];
s q[0];
s q[64];
s q[5];
s q[3];
cx q[27],q[44];
h q[34];
h q[20];
h q[35];
s q[10];
cx q[48],q[33];
s q[20];
s q[53];
h q[46];
h q[35];
cx q[1],q[16];
cx q[21],q[47];
s q[57];
cx q[58],q[50];
h q[21];
s q[9];
cx q[45],q[62];
cx q[44],q[8];
cx q[24],q[39];
s q[8];
cx q[30],q[26];
cx q[32],q[59];
s q[14];
cx q[39],q[34];
cx q[15],q[48];
h q[58];
s q[48];
s q[44];
s q[2];
h q[22];
cx q[41],q[30];
h q[47];
h q[47];
s q[13];
cx q[58],q[59];
s q[57];
s q[39];
cx q[22],q[5];
s q[48];
h q[51];
h q[63];
cx q[50],q[36];
h q[50];
h q[29];
cx q[30],q[17];
h q[41];
s q[11];
cx q[56],q[29];
s q[43];
s q[30];
h q[38];
cx q[12],q[38];
s q[37];
cx q[17],q[46];
s q[9];
cx q[9],q[35];
s q[50];
cx q[48],q[33];
h q[22];
s q[64];
cx q[48],q[58];
s q[14];
h q[48];
s q[62];
h q[25];
s q[9];
s q[61];
s q[49];
h q[46];
s q[34];
s q[54];
s q[59];
s q[2];
s q[21];
h q[52];
cx q[33],q[3];
cx q[1],q[36];
s q[23];
s q[43];
cx q[13],q[63];
cx q[26],q[17];
cx q[10],q[55];
h q[25];
h q[0];
cx q[7],q[56];
h q[52];
s q[59];
h q[28];
s q[31];
h q[30];
cx q[36],q[43];
s q[62];
s q[25];
h q[58];
s q[27];
h q[20];
h q[58];
h q[4];
cx q[48],q[57];
cx q[20],q[43];
h q[23];
cx q[12],q[61];
cx q[29],q[37];
s q[40];
h q[6];
h q[24];
s q[39];
s q[12];
h q[55];
cx q[8],q[62];
cx q[53],q[5];
h q[1];
cx q[57],q[20];
h q[6];
h q[39];
cx q[14],q[42];
cx q[22],q[6];
s q[62];
cx q[54],q[58];
h q[7];
cx q[1],q[24];
cx q[30],q[48];
cx q[31],q[25];
s q[44];
h q[38];
cx q[27],q[20];
cx q[54],q[41];
h q[58];
cx q[64],q[3];
h q[45];
h q[57];
h q[15];
h q[42];
cx q[20],q[33];
h q[57];
s q[46];
cx q[12],q[53];
s q[54];
cx q[55],q[25];
s q[3];
h q[54];
cx q[17],q[47];
s q[45];
cx q[16],q[53];
h q[44];
h q[2];
s q[12];
s q[44];
s q[54];
cx q[63],q[0];
s q[64];
s q[30];
cx q[34],q[10];
s q[41];
h q[26];
cx q[21],q[16];
cx q[19],q[31];
cx q[28],q[27];
h q[63];
h q[17];
s q[58];
s q[9];
h q[4];
cx q[63],q[18];
h q[54];
cx q[32],q[26];
h q[5];
s q[10];
s q[55];
cx q[63],q[25];
s q[12];
cx q[41],q[48];
cx q[56],q[50];
cx q[36],q[40];
s q[0];
h q[21];
cx q[22],q[26];
h q[7];
h q[38];
cx q[62],q[30];
cx q[39],q[11];
s q[47];
s q[55];
s q[48];
cx q[58],q[52];
h q[12];
cx q[64],q[11];
s q[52];
s q[8];
cx q[50],q[18];
h q[12];
h q[57];
cx q[0],q[60];
cx q[52],q[36];
s q[58];
s q[8];
cx q[64],q[31];
cx q[31],q[7];
cx q[45],q[3];
s q[61];
s q[20];
h q[12];
s q[42];
h q[8];
s q[19];
s q[39];
h q[0];
cx q[26],q[44];
h q[4];
cx q[15],q[28];
h q[55];
cx q[56],q[47];
s q[17];
cx q[48],q[21];
cx q[38],q[50];
s q[37];
cx q[20],q[4];
h q[34];
s q[18];
h q[27];
s q[43];
cx q[6],q[23];
cx q[48],q[3];
s q[64];
s q[44];
s q[35];
cx q[21],q[26];
h q[17];
s q[42];
cx q[2],q[35];
h q[53];
cx q[6],q[18];
h q[3];
h q[64];
s q[31];
h q[59];
s q[51];
s q[25];
h q[12];
s q[41];
s q[6];
h q[5];
cx q[42],q[58];
s q[34];
cx q[3],q[38];
s q[56];
h q[38];
cx q[1],q[55];
cx q[21],q[63];
s q[3];
cx q[62],q[53];
s q[43];
h q[51];
h q[5];
s q[23];
cx q[5],q[31];
h q[36];
h q[31];
h q[23];
s q[21];
h q[55];
h q[39];
cx q[34],q[14];
s q[47];
s q[21];
h q[64];
s q[48];
h q[64];
s q[2];
h q[2];
s q[25];
cx q[43],q[49];
cx q[30],q[62];
cx q[31],q[33];
cx q[41],q[20];
cx q[56],q[15];
cx q[31],q[22];
h q[43];
h q[25];
s q[40];
s q[49];
cx q[58],q[46];
cx q[22],q[55];
h q[55];
cx q[10],q[38];
s q[50];
cx q[15],q[60];
s q[4];
h q[11];
s q[34];
cx q[37],q[61];
s q[57];
s q[12];
h q[16];
s q[45];
h q[26];
s q[45];
h q[29];
cx q[46],q[39];
cx q[20],q[51];
s q[29];
s q[21];
cx q[25],q[58];
cx q[25],q[50];
h q[3];
cx q[1],q[3];
h q[35];
h q[51];
cx q[38],q[49];
h q[48];
cx q[55],q[21];
h q[45];
s q[62];
s q[51];
h q[8];
s q[3];
h q[36];
