#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdlib.h>
#include <vector_types.h>
#include <iostream>
#include <ostream>

#include "JpegCommon.h"
#include "JpegCPU.h"
#include "JpegDimension.cpp"
#include "JpegGPU.h"

using namespace std;

int main() {
    int szx = 0, szy = 0;
    int **matrixTmp = NULL;
    matrixTmp = pgmOpen((char *) "i.pgm", szx, szy);
    int matrix[szy][szx];
    for (int j = 0; j < szy; ++j) {
        for (int i = 0; i < szx; ++i) {
            matrix[j][i] = matrixTmp[j][i];
        }
    }
    /*define block size and length of zigzag matrix*/
    int LEN = BLOCK * BLOCK * PERCENT / 100;
    const int len = (LEN > BLOCK * BLOCK ? BLOCK * BLOCK : LEN);

    /*zigzag index encoding array*/
    int *Encodearr = (int *) malloc(2 * len * sizeof(int));
    zigzagEncodeArray(Encodearr, len);

    /*device encode array*/
    int *d_Encodearr = NULL;
    errCUDA(cudaMalloc((void **) &d_Encodearr, 2 * len * sizeof(int)));
    errCUDA(cudaMemcpy(d_Encodearr, Encodearr, 2 * len * sizeof(int), cudaMemcpyHostToDevice));

    /*device image matrix*/
    int *d_matrix = NULL, *d_Qmatrix8;
    errCUDA(cudaMalloc((void **) &d_matrix, szx * szy * sizeof(int *)));
    errCUDA(cudaMemcpy(d_matrix, matrix, szx * szy * sizeof(int), cudaMemcpyHostToDevice));

    /*JPEG ISO quantize matrix*/
    errCUDA(cudaMalloc((void **) &d_Qmatrix8, BLOCK * BLOCK * sizeof(int *)));
    errCUDA(cudaMemcpy(d_Qmatrix8, Qmatrix8, BLOCK * BLOCK * sizeof(int), cudaMemcpyHostToDevice));

    /*define JPEG blocks*/
    int blockX = (szx + BLOCK - 1) / BLOCK, blockY = (szy + BLOCK - 1) / BLOCK;
    dim3 block(BLOCK, BLOCK);
    dim3 grid(blockX, blockY);

    /*device zigzag array*/
    int *d_outarr = NULL;
    errCUDA(cudaMalloc((void **) &d_outarr, blockX * blockY * len * sizeof(int)));

    /*zigzag array memory*/
    int *outarr = (int *) malloc(blockX * blockY * len * sizeof(int));

    /*keep time now*/
    clock_t tcpu0, tcpu, tgpu0, tgpu;
    /*main CPU function compare*/
    tcpu0 = clock();
    CPU_MAIN(matrixTmp, Qmatrix8, outarr, Encodearr, blockX, blockY, szx, szy, len);
    tcpu = clock() - tcpu0;
    /*write image file*/
    pgmWrite((char *) "ocpu.pgm", matrixTmp, szx, szy);

    /*main GPU cuda function*/
    tgpu0 = clock();
    CUDA_MAIN<<<grid, block>>>(d_matrix, d_outarr, d_Qmatrix8, d_Encodearr, len, blockX, szx, szy);
    tgpu = clock() - tgpu0;

    /*test: print zigzag array*/
    errCUDA(cudaMemcpy(outarr, d_outarr, blockX * blockY * len * sizeof(int), cudaMemcpyDeviceToHost));

    /*output matrix*/
    int matrix2[szy][szx];
    errCUDA(cudaMemcpy(matrix2, d_matrix, szx * szy * sizeof(int), cudaMemcpyDeviceToHost));
    for (int j = 0; j < szy; ++j) {
        for (int i = 0; i < szx; ++i) {
            matrixTmp[j][i] = saturation8bit(matrix2[j][i]);
        }
    }

    /*write image file*/
    pgmWrite((char *) "ogpu.pgm", matrixTmp, szx, szy);

    /*destroy*/
    errCUDA(cudaFree(d_Encodearr));
    errCUDA(cudaFree(d_Qmatrix8));
    errCUDA(cudaFree(d_matrix));
    errCUDA(cudaFree(d_outarr));
    errCUDA(cudaDeviceReset());

    /*result report*/
    int faster = tcpu / tgpu;
    if (faster > 100) {
        cout << "!!Breaking News!!" << endl;
        cout << "My GPU is " << faster << " times faster than CPU!" << endl;
    }
    if (faster < 1) {
        cout << "GPU is a crap!!" << endl;
    }
    cout << endl << "Details:" << endl << "GPU: " << tgpu << " ms" << endl;
    cout << "CPU: " << tcpu << " ms" << endl;

    return 0;
}
