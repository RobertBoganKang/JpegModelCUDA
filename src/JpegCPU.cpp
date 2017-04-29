#include <device_functions.hpp>
#include <math.h>
#include <math_functions_dbl_ptx3.hpp>
#include <stdlib.h>

#include "JpegCommon.h"
#include "JpegDimension.cpp"

using namespace std;

float SsumMatrix(float matrix[BLOCK][BLOCK]) {
    float count = 0;
    for (int i = 0; i < BLOCK; ++i) {
        for (int j = 0; j < BLOCK; ++j) {
            count += matrix[i][j];
        }
    }
    return count;
}

int Sedges(int **matrix, int idx, int idy, int IMGX, int IMGY) {
    int res;
    res = matrix[idy][idx];
    if (idx > IMGX - 1 && idy > IMGY - 1)
        res = matrix[(IMGY - 1)][(IMGX - 1)];
    if (idx > IMGX - 1 && idy <= IMGY - 1)
        res = matrix[idy][(IMGX - 1)];
    if (idx <= IMGX - 1 && idy > IMGY - 1)
        res = matrix[(IMGY - 1)][idx];
    return res;
}

void SfourierDCT_zigzagEncode(int **matrix, int Qmatrix8[][BLOCK], int *arr, int *idxarr, int szx, int szy, int len) {
    int clen = 0;
    /*fourierDCT part*/
    for (int my = 0; my < szy; my += BLOCK) {
        for (int mx = 0; mx < szx; mx += BLOCK) {
            float tempMatrix2[BLOCK][BLOCK];
            for (int v = 0; v < BLOCK; ++v) {
                for (int u = 0; u < BLOCK; ++u) {
                    float tempMatrix[BLOCK][BLOCK];
                    float alpha = (float) ((u == 0 || v == 0) ? (u == 0 && v == 0 ? (float) (1.0 / 2.0) : sqrtf(
                            (float) (1.0 / 2.0))) : 1.0);
                    for (int y = 0; y < BLOCK; ++y) {
                        for (int x = 0; x < BLOCK; ++x) {
                            /*matrix edge handling is needed*/
                            tempMatrix[y][x] = (float) ((Sedges(matrix, mx + x, my + y, szx, szy) - 128) *
                                                        cos((2 * x + 1) * u * M_PI / 16) *
                                                        cos((2 * y + 1) * v * M_PI / 16));
                        }
                    }
                    tempMatrix2[v][u] = alpha * SsumMatrix(tempMatrix);
                }
            }
            /*zigzag encode part*/
            for (int i = 0; i < len; ++i) {
                int u = idxarr[2 * i];
                int v = idxarr[2 * i + 1];
                arr[clen + i] = (int) round(1.0 / 4.0 * tempMatrix2[v][u] / Qmatrix8[v][u]);
            }
            clen += len;
        }
    }
}

void Szeros_zigzagDecode_inversefourierDCT(int **matrix, int Qmatrix8[][BLOCK], int *arr, int *idxarr, int szx, int szy,
                                           int len) {
    int count = 0;
    for (int my = 0; my < szy; my += BLOCK) {
        for (int mx = 0; mx < szx; mx += BLOCK) {
            /*zeros*/
            for (int i = 0; i < BLOCK; ++i) {
                for (int j = 0; j < BLOCK; ++j) {
                    matrix[my + j][mx + i] = 0;
                }
            }
            /*zigzag decode*/
            for (int k = 0; k < len; ++k) {
                int u = idxarr[2 * k];
                int v = idxarr[2 * k + 1];
                matrix[my + v][mx + u] = arr[count++];
            }
            /*fourierDCT*/
            float tempMatrix2[BLOCK][BLOCK];
            for (int y = 0; y < BLOCK; ++y) {
                for (int x = 0; x < BLOCK; ++x) {
                    float tempMatrix[BLOCK][BLOCK];
                    for (int v = 0; v < BLOCK; ++v) {
                        for (int u = 0; u < BLOCK; ++u) {
                            float alpha = (float) ((u == 0 || v == 0) ? (u == 0 && v == 0 ? (float) (1.0 / 2.0) : sqrtf(
                                    (float) (1.0 / 2.0))) : 1.0);
                            tempMatrix[v][u] = (float) (alpha * matrix[my + v][mx + u] * Qmatrix8[v][u] *
                                                        cos((2 * x + 1) * u * M_PI / 16) *
                                                        cos((2 * y + 1) * v * M_PI / 16));
                        }
                    }
                    tempMatrix2[y][x] = SsumMatrix(tempMatrix);
                }
            }
            for (int y = 0; y < BLOCK; ++y) {
                for (int x = 0; x < BLOCK; ++x) {
                    matrix[my + y][mx + x] = (int) round(1.0 / 4.0 * tempMatrix2[y][x]) + 127;
                }
            }
        }
    }
}

void
CPU_MAIN(int **matrixTmp, int Qmatrix8[][BLOCK], int *outarr, int *Encodearr, int blockX, int blockY, int szx, int szy,
         int len) {
    SfourierDCT_zigzagEncode(matrixTmp, Qmatrix8, outarr, Encodearr, szx, szy, len);

    /*de-zigzag matrix*/
    int **midMatrix = (int **) malloc(blockY * BLOCK * sizeof(int *));
    for (int i = 0; i < blockY * BLOCK; ++i) {
        midMatrix[i] = (int *) malloc(blockX * BLOCK * sizeof(int));
    }

    /*same gap here
     * as GPU
     * ...*/

    Szeros_zigzagDecode_inversefourierDCT(midMatrix, Qmatrix8, outarr, Encodearr, szx, szy, len);

    /*print matrix*/
    for (int j = 0; j < szy; ++j) {
        for (int i = 0; i < szx; ++i) {
            matrixTmp[j][i] = saturation8bit(midMatrix[j][i]);
        }
    }
}
