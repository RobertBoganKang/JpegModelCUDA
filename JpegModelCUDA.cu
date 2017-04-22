#include <iostream>
#include <math.h>
#include <fstream>
#include <cuda_runtime.h>

using namespace std;

#define BLOCK 8
#define PERCENT 30

#define IMGX 16
#define IMGY 16

/*CPU Part*/
/*JPEG standard quantization matrix for brightness @(8 * 8)*/
int Qmatrix8[BLOCK][BLOCK] = {{16, 11, 10, 16, 24,  40,  51,  61},
                              {12, 12, 14, 19, 26,  58,  60,  55},
                              {14, 13, 16, 24, 40,  57,  69,  56},
                              {14, 17, 22, 29, 51,  87,  80,  62},
                              {18, 22, 37, 56, 68,  109, 103, 77},
                              {24, 35, 55, 64, 81,  104, 113, 92},
                              {49, 64, 78, 87, 103, 121, 120, 101},
                              {72, 92, 95, 98, 112, 100, 103, 99}};

int matrix[IMGY][IMGX] = {{255, 255, 255, 255, 222, 125, 58,  22,  8,   43,  85,  203, 255, 255, 255, 255},
                          {255, 255, 222, 89,  2,   0,   0,   0,   0,   0,   0,   2,   137, 255, 255, 255},
                          {255, 255, 233, 15,  9,   100, 161, 208, 210, 155, 26,  0,   4,   211, 255, 255},
                          {255, 255, 255, 196, 235, 255, 255, 255, 255, 255, 207, 1,   0,   106, 255, 255},
                          {255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 235, 31,  0,   66,  255, 255},
                          {255, 255, 255, 255, 255, 216, 135, 76,  44,  11,  0,   0,   0,   33,  255, 255},
                          {255, 255, 255, 179, 43,  0,   0,   14,  64,  96,  127, 33,  0,   31,  255, 255},
                          {255, 255, 171, 0,   0,   62,  190, 250, 255, 255, 255, 59,  0,   31,  255, 255},
                          {255, 255, 45,  0,   43,  251, 255, 255, 255, 255, 255, 59,  0,   31,  255, 255},
                          {255, 255, 18,  0,   81,  255, 255, 255, 255, 255, 193, 22,  0,   31,  255, 255},
                          {255, 255, 68,  0,   4,   143, 225, 224, 172, 87,  0,   0,   0,   31,  255, 255},
                          {255, 255, 205, 9,   0,   0,   0,   0,   0,   2,   102, 117, 0,   31,  255, 255},
                          {255, 255, 255, 215, 91,  32,  7,   32,  99,  208, 255, 165, 0,   31,  255, 255},
                          {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
                          {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
                          {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255}};

void zigzagEncodeArray(int *array, int len) {
    bool backwards = false;
    int arrIdx = 0;
    for (int i = 0; i < BLOCK; ++i) {
        for (int j = 0; j < i + 1; ++j) {
            if (arrIdx >= 2 * len)
                break;
            array[arrIdx++] = (backwards ? j : i - j);
            array[arrIdx++] = (backwards ? i - j : j);
        }
        backwards = !backwards;
    }
    if (len > 1 / 2 * BLOCK * (BLOCK + 1)) {
        for (int i = BLOCK - 1; i > 0; i--) {
            for (int j = 0; j < i; ++j) {
                if (arrIdx >= 2 * len)
                    break;
                array[arrIdx++] = (backwards ? BLOCK - i + j : BLOCK - j - 1);
                array[arrIdx++] = (backwards ? BLOCK - j - 1 : BLOCK - i + j);
            }
            backwards = !backwards;
        }
    }
}

/**encode part*/
__device__ float SumMatrix(float *matrix, int row, int col) {
    for (int offset = 1; offset < BLOCK; offset *= 2) {
        if ((row % (2 * offset) == 0 && col % (2 * offset) == 0) && row < BLOCK
            && col < BLOCK) {
            matrix[row * BLOCK + col] += (matrix[(row) * BLOCK + (col + offset)]
                                          + matrix[(row + offset) * BLOCK + (col)]
                                          + matrix[(row + offset) * BLOCK + (col + offset)]);
        }
        __syncthreads();
    }
    return matrix[0];
}

__device__ void FourierDCT(int *matrix, int *omatrix, int *Qmatrix8, int idx,
                           int idy, int row, int col) {
    __shared__ float tempmtr[BLOCK * BLOCK], tempmtr2[BLOCK * BLOCK];
    __shared__ int matrixS[BLOCK * BLOCK];
    matrixS[row * BLOCK + col] = matrix[idx * IMGX + idy] - 128;
    __syncthreads();
    for (int v = 0; v < BLOCK; v++) {
        for (int u = 0; u < BLOCK; u++) {
            tempmtr[row * BLOCK + col] = (float) (matrixS[row * BLOCK + col]
                                                  * cos((2 * row + 1) * u * M_PI / 16)
                                                  * cos((2 * col + 1) * v * M_PI / 16));
            __syncthreads();
            float alpha = (float) (
                    (u == 0 || v == 0) ?
                    (u == 0 && v == 0 ?
                     (float) (1.0 / 2.0) :
                     sqrtf((float) (1.0 / 2.0))) :
                    1.0);
            tempmtr2[u * BLOCK + v] = 1.0 / 4.0 * alpha
                                      * SumMatrix(tempmtr, row, col);
        }
    }
    omatrix[row * BLOCK + col] = round(
            tempmtr2[row * BLOCK + col] / Qmatrix8[row * BLOCK + col]);
    __syncthreads();
    /*test*/
    matrix[idx * IMGX + idy] = omatrix[row * BLOCK + col];
}

__device__ void ZigzagEncode(int *matrix, int *array, int *idxarr, int len,
                             int arrIdx, int row, int col) {
    for (int i = 0; i < len; ++i) {
        int x = idxarr[2 * i];
        int y = idxarr[2 * i + 1];
        if (row == x && col == y)
            array[arrIdx + i] = matrix[x * BLOCK + y];
    }
}

/**decode part*/
__device__ void Zeros(int *matrix, int row, int col) {
    matrix[row * BLOCK + col] = 0;
    __syncthreads();
}

__device__ void ZigzagDecode(int *matrix, int *array, int *idxarr, int len,
                             int arrIdx, int row, int col) {
    for (int i = 0; i < len; ++i) {
        int x = idxarr[2 * i];
        int y = idxarr[2 * i + 1];
        if (row == x && col == y)
            matrix[x * BLOCK + y] = array[i + arrIdx];
    }
}

__device__ void InverseFourierDCT(int *matrix, int *Qmatrix8, int idx, int idy,
                                  int row, int col) {
    __shared__ float tempmtr[BLOCK * BLOCK], tempmtr2[BLOCK * BLOCK];
    __shared__ int matrixS[BLOCK * BLOCK];
    matrixS[row * BLOCK + col] = matrix[idx * IMGX + idy]
                                 * Qmatrix8[row * BLOCK + col];
    __syncthreads();
    for (int y = 0; y < BLOCK; y++) {
        for (int x = 0; x < BLOCK; x++) {
            float alpha = (float) (
                    (row == 0 || col == 0) ?
                    (row == 0 && col == 0 ?
                     (float) (1.0 / 2.0) :
                     sqrtf((float) (1.0 / 2.0))) :
                    1.0);
            tempmtr[row * BLOCK + col] = (float) (alpha
                                                  * matrixS[row * BLOCK + col]
                                                  * cos((2 * x + 1) * row * M_PI / 16)
                                                  * cos((2 * y + 1) * col * M_PI / 16));
            __syncthreads();
            tempmtr2[x * BLOCK + y] = SumMatrix(tempmtr, row, col);
            __syncthreads();
        }
    }
    matrix[idx * IMGX + idy] = round(
            1.0 / 4.0 * tempmtr2[row * BLOCK + col] + 127);
    __syncthreads();
}

__global__ void CUDA_MAIN(int *matrix, int *oarr, int *Qmatrix8, int *idxarr,
                          int len, int blockX) {
    int row = threadIdx.x, col = threadIdx.y;
    int idx = blockIdx.x * blockDim.x + row, idy = blockIdx.y * blockDim.y
                                                   + col;
    int arrIdx = len * (blockIdx.x * blockX + blockIdx.y);
    __shared__ int omatrix[BLOCK * BLOCK];
    /**encoder*/
    FourierDCT(matrix, omatrix, Qmatrix8, idx, idy, row, col);
    ZigzagEncode(omatrix, oarr, idxarr, len, arrIdx, row, col);

 	/*break point here for
	 *  array handeling
	 *  ...*/
  
    /**decoder*/
    Zeros(omatrix, row, col);
    ZigzagDecode(omatrix, oarr, idxarr, len, arrIdx, row, col);
    matrix[idx * IMGX + idy] = omatrix[row * BLOCK + col];
    InverseFourierDCT(matrix, Qmatrix8, idx, idy, row, col);
}

int main() {
    /*define block size and length of zigzag matrix*/
    const int szx = IMGX;
    const int szy = IMGY;
    int LEN = BLOCK * BLOCK * PERCENT / 100;
    const int len = (LEN > BLOCK * BLOCK ? BLOCK * BLOCK : LEN);

    /**encoding part*/
    int *Encodearr = (int *) malloc(2 * len * sizeof(int));
    zigzagEncodeArray(Encodearr, len);

    /*print zigzag array*/
//	cout << "Zigzag index array with length: " << len << endl;
//	for (int i = 0; i < 2 * len; ++i) {
//		cout << Encodearr[i] << " ";
//	}
//	cout << endl;
    int *d_Encodearr = NULL;
    cudaMalloc((void **) &d_Encodearr, 2 * len * sizeof(int));
    cudaMemcpy(d_Encodearr, Encodearr, 2 * len * sizeof(int),
               cudaMemcpyHostToDevice);

    /*copy image matrix*/
    int *d_matrix = NULL, *d_Qmatrix8;
    cudaMalloc((void **) &d_matrix, szx * szy * sizeof(int *));
    cudaMemcpy(d_matrix, matrix, szx * szy * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_Qmatrix8, BLOCK * BLOCK * sizeof(int *));
    cudaMemcpy(d_Qmatrix8, Qmatrix8, BLOCK * BLOCK * sizeof(int),
               cudaMemcpyHostToDevice);

    /**decoding part*/
    int blockX = (IMGX + BLOCK - 1) / BLOCK,
            blockY = (IMGY + BLOCK - 1) / BLOCK;
    dim3 block(BLOCK, BLOCK);
    dim3 grid(blockX, blockY);
    /*get output of zigzag array*/
    int *d_outarr = NULL;
    cudaMalloc((void **) &d_outarr, blockX * blockY * len * sizeof(int));
    /*restore zigzag matrix*/
    int *d_outmtr = NULL;
    cudaMalloc((void **) &d_outmtr, szx * szy * sizeof(int *));
    CUDA_MAIN<<<grid, block>>>(d_matrix, d_outarr, d_Qmatrix8, d_Encodearr, len,
            blockX);

    /*test: print zigzag array index*/
    int *outarr = (int *) malloc(blockX * blockY * len * sizeof(int));
    cudaMemcpy(outarr, d_outarr, blockX * blockY * len * sizeof(int),
               cudaMemcpyDeviceToHost);

    /*output matrix*/
    int matrix2[szy][szx];
    cudaMemcpy(matrix2, d_matrix, szx * szy * sizeof(int),
               cudaMemcpyDeviceToHost);

    /*print the zigzag array*/
    cout << "Output Zigzag Array:" << endl;
    for (int i = 0; i < blockX * blockY * len; i++) {
        cout << outarr[i] << "\t";
    }
    cout << endl;
    /*print the matrix*/
    cout << "Output Matrix @a.dat" << endl;
    for (int i = 0; i < szy; i++) {
        for (int j = 0; j < szx; ++j) {
            cout << matrix2[i][j];
            cout << "\t";
        }
        cout << endl;
    }
    ofstream myfile;
    myfile.open("a.dat");
    myfile << "{{";
    for (int i = 0; i < szy; i++) {
        for (int j = 0; j < szx; ++j) {
            myfile << matrix2[i][j];
            if (j != szx - 1)
                myfile << ",";
        }
        if (i != szy - 1)
            myfile << "},{";
        else
            myfile << "}}";
    }
    myfile.close();
    return 0;
}

