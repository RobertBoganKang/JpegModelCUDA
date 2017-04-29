#ifndef SOLUTION_JPEGCPU_H
#define SOLUTION_JPEGCPU_H

#include "JpegDimension.cpp"

int saturation8bit(int i);

void zigzagEncodeArray(int *array, int len);

float SsumMatrix(float matrix[BLOCK][BLOCK]);

int Sedges(int *matrix, int idx, int idy, int IMGX, int IMGY);

void SfourierDCT_zigzagEncode(int **matrix, int Qmatrix8[][BLOCK], int *arr, int *idxarr, int szx, int szy, int len);

void
Szeros_zigzagDecode_inversefourierDCT(int **matrix, int Qmatrix8[][BLOCK], int *arr, int *idxarr, int szx, int szy,
                                      int len);

void
CPU_MAIN(int **matrixTmp, int Qmatrix8[][BLOCK], int *outarr, int *Encodearr, int blockX, int blockY, int szx, int szy,
         int len);

#endif //SOLUTION_JPEGCPU_H
