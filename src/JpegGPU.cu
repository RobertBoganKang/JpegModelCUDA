#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <device_functions.hpp>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <host_defines.h>
#include <math_functions.hpp>
#include <math_functions_dbl_ptx3.hpp>
#include <stdio.h>
#include <vector_types.h>

#include "JpegDimension.cpp"

using namespace std;

/**error handling simple*/
/*global counter*/
int counter = 0;

void errCUDA(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error position %d (error code %s)!\n", counter, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    counter++;
}

/**GPU part*/
/*encode part*/
__device__ float SumMatrix(float *matrix, int row, int col) {
    for (int offset = 1; offset < BLOCK; offset *= 2) {
        if ((row % (2 * offset) == 0 && col % (2 * offset) == 0) && row < BLOCK && col < BLOCK) {
            matrix[row * BLOCK + col] += (matrix[(row) * BLOCK + (col + offset)] +
                                          matrix[(row + offset) * BLOCK + (col)] +
                                          matrix[(row + offset) * BLOCK + (col + offset)]);
        }
        __syncthreads();
    }
    return matrix[0];
}

__device__ int Edges(int *matrix, int idx, int idy, int IMGX, int IMGY) {
    int res;
    res = matrix[idx * IMGX + idy];
    if (idx > IMGX - 1 && idy > IMGY - 1)
        res = matrix[(IMGX - 1) * IMGX + (IMGY - 1)];
    if (idx > IMGX - 1 && idy <= IMGY - 1)
        res = matrix[(IMGX - 1) * IMGX + idy];
    if (idx <= IMGX - 1 && idy > IMGY - 1)
        res = matrix[idx * IMGX + (IMGY - 1)];
    return res;
}

__device__ void
FourierDCT(int *matrix, int *omatrix, int *Qmatrix8, int idx, int idy, int row, int col, int IMGX, int IMGY) {
    __shared__ float tempmtr[BLOCK * BLOCK], tempmtr2[BLOCK * BLOCK];
    __shared__ int matrixS[BLOCK * BLOCK];
    /*the matrix needs edge handling for dimensions not the multiple of 8*/
    matrixS[row * BLOCK + col] = Edges(matrix, idx, idy, IMGX, IMGY) - 128;
    __syncthreads();
    for (int v = 0; v < BLOCK; v++) {
        for (int u = 0; u < BLOCK; u++) {
            tempmtr[row * BLOCK + col] = (float) (matrixS[row * BLOCK + col] * cos((2 * row + 1) * u * M_PI / 16) *
                                                  cos((2 * col + 1) * v * M_PI / 16));
            __syncthreads();
            float alpha = (float) ((u == 0 || v == 0) ? (u == 0 && v == 0 ? (float) (1.0 / 2.0) : sqrtf(
                    (float) (1.0 / 2.0))) : 1.0);
            tempmtr2[u * BLOCK + v] = (float) (1.0 / 4.0 * alpha * SumMatrix(tempmtr, row, col));
        }
    }
    omatrix[row * BLOCK + col] = (int) round(tempmtr2[row * BLOCK + col] / Qmatrix8[row * BLOCK + col]);
    __syncthreads();
    matrix[idx * IMGX + idy] = omatrix[row * BLOCK + col];
}

__device__ void ZigzagEncode(int *matrix, int *array, int *idxarr, int len, int arrIdx, int row, int col) {
    int i = row * BLOCK + col;
    if (i < len) {
        int x = idxarr[2 * i];
        int y = idxarr[2 * i + 1];
        array[arrIdx + i] = matrix[x * BLOCK + y];
    }
}

/*decode part*/
__device__ void Zeros(int *matrix, int row, int col) {
    matrix[row * BLOCK + col] = 0;
    __syncthreads();
}

__device__ void ZigzagDecode(int *matrix, int *array, int *idxarr, int len, int arrIdx, int row, int col) {
    int i = row * BLOCK + col;
    if (i < len) {
        int x = idxarr[2 * i];
        int y = idxarr[2 * i + 1];
        matrix[x * BLOCK + y] = array[i + arrIdx];
    }
}

__device__ void InverseFourierDCT(int *matrix, int *Qmatrix8, int idx, int idy, int row, int col, int IMGX) {
    __shared__ float tempmtr[BLOCK * BLOCK], tempmtr2[BLOCK * BLOCK];
    __shared__ int matrixS[BLOCK * BLOCK];
    matrixS[row * BLOCK + col] = matrix[idx * IMGX + idy] * Qmatrix8[row * BLOCK + col];
    __syncthreads();
    for (int y = 0; y < BLOCK; y++) {
        for (int x = 0; x < BLOCK; x++) {
            float alpha = (float) ((row == 0 || col == 0) ? (row == 0 && col == 0 ? (float) (1.0 / 2.0) : sqrtf(
                    (float) (1.0 / 2.0))) : 1.0);
            tempmtr[row * BLOCK + col] = (float) (alpha * matrixS[row * BLOCK + col] *
                                                  cos((2 * x + 1) * row * M_PI / 16) *
                                                  cos((2 * y + 1) * col * M_PI / 16));
            __syncthreads();
            tempmtr2[x * BLOCK + y] = SumMatrix(tempmtr, row, col);
            __syncthreads();
        }
    }
    matrix[idx * IMGX + idy] = (int) round(1.0 / 4.0 * tempmtr2[row * BLOCK + col] + 127);
    __syncthreads();
}

/*GPU operations*/
__global__ void CUDA_MAIN(int *matrix, int *oarr, int *Qmatrix8, int *idxarr, int len, int blockX, int IMGX, int IMGY) {
    int row = threadIdx.x, col = threadIdx.y;
    int idx = blockIdx.x * blockDim.x + row, idy = blockIdx.y * blockDim.y + col;
    int arrIdx = len * (blockIdx.x * blockX + blockIdx.y);
    __shared__ int omatrix[BLOCK * BLOCK];
    /**encoder*/
    FourierDCT(matrix, omatrix, Qmatrix8, idx, idy, row, col, IMGX, IMGY);
    ZigzagEncode(omatrix, oarr, idxarr, len, arrIdx, row, col);

    /*Real program will divide into two parts,
     * where encoder and decoder are there.
     * The Huffman compression algorithm will be implemented at CPU sequentially,
     * and JPEG write and read processes are also here.
     * Since this is only the model of JPEG algorithm with parallel computing,
     * I leave the gap here for future work if necessary*/

    /**decoder*/
    Zeros(omatrix, row, col);
    ZigzagDecode(omatrix, oarr, idxarr, len, arrIdx, row, col);
    matrix[idx * IMGX + idy] = omatrix[row * BLOCK + col];
    InverseFourierDCT(matrix, Qmatrix8, idx, idy, row, col, IMGX);
}
