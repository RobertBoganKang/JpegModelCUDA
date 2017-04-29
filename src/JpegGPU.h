#ifndef JPEGGPU_H_
#define JPEGGPU_H_

#include <driver_types.h>

/**error handling simple*/
/*global counter*/
extern int counter;

void errCUDA(cudaError_t err);

/**GPU part*/
/*encode part*/
__device__ float SumMatrix(float *matrix, int row, int col);

__device__ int Edges(int *matrix, int idx, int idy, int IMGX, int IMGY);

__device__ void
FourierDCT(int *matrix, int *omatrix, int *Qmatrix8, int idx, int idy, int row, int col, int IMGX, int IMGY);

__device__ void ZigzagEncode(int *matrix, int *array, int *idxarr, int len, int arrIdx, int row, int col);

/*decode part*/
__device__ void Zeros(int *matrix, int row, int col);

__device__ void ZigzagDecode(int *matrix, int *array, int *idxarr, int len, int arrIdx, int row, int col);

__device__ void InverseFourierDCT(int *matrix, int *Qmatrix8, int idx, int idy, int row, int col, int IMGX);
/*GPU operations*/
__global__ void CUDA_MAIN(int *matrix, int *oarr, int *Qmatrix8, int *idxarr, int len, int blockX, int IMGX, int IMGY);

#endif /* JPEGGPU_H_ */
