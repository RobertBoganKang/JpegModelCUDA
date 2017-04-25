#include <iostream>
#include <math.h>
#include <fstream>
#include <cuda_runtime.h>

using namespace std;

#define BLOCK 8
#define PERCENT 30

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
/**image Read and Write*/
/*this function is only available in c++11*/
int stoi(char *c) {
    int sum = 0, sign = 1;
    for (int i = 0; i < 4; i++) {
        if (c[i] <= '9' && c[i] >= '0') {
            sum *= 10;
            sum += (c[i] - '0');
        }
        if (c[i] == '-')
            sign = -1;
    }
    sum *= sign;
    return sum;
}

int **pgmOpen(char *file, int &imgx, int &imgy) {
    FILE *fp = fopen(file, "r");
    char str[4];
    fscanf(fp, "%s", str);
    if (str[0] != 'P' && str[1] != '2') {
        cout << "Error Pics!";
    }
    fscanf(fp, "%s", str);
    imgx = stoi(str);
    fscanf(fp, "%s", str);
    imgy = stoi(str);
    fscanf(fp, "%s", str);
    int **imgdata = (int **) malloc(imgx * sizeof(int *));
    for (int i = 0; i < imgy; ++i) {
        imgdata[i] = (int *) malloc(imgy * sizeof(int));
    }
    for (int i = 0; i < imgx; ++i) {
        for (int j = 0; j < imgy; ++j) {
            fscanf(fp, "%s", str);
            imgdata[i][j] = stoi(str);
        }
    }
    fclose(fp);
    return imgdata;
}

void pgmWrite(char *file, int **imgdata, int imgx, int imgy) {
    ofstream myfile;
    myfile.open(file);
    myfile << "P2\n";
    myfile << imgx << " " << imgy << "\n" << "255" << endl;
    for (int i = 0; i < imgx; ++i) {
        for (int j = 0; j < imgy; ++j) {
            myfile << imgdata[i][j] << "\t";
        }
        myfile << "\n";
    }
    myfile.close();
}

/**CPU Part*/
/*JPEG standard quantization matrix for brightness @(8 * 8)*/
int Qmatrix8[BLOCK][BLOCK] = {{16, 11, 10, 16, 24,  40,  51,  61},
                              {12, 12, 14, 19, 26,  58,  60,  55},
                              {14, 13, 16, 24, 40,  57,  69,  56},
                              {14, 17, 22, 29, 51,  87,  80,  62},
                              {18, 22, 37, 56, 68,  109, 103, 77},
                              {24, 35, 55, 64, 81,  104, 113, 92},
                              {49, 64, 78, 87, 103, 121, 120, 101},
                              {72, 92, 95, 98, 112, 100, 103, 99}};

int saturation8bit(int i) {
    return (i < 0 ? 0 : (i > 255 ? 255 : i));
}

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

int main() {
    int szx = 0, szy = 0;
    int **matrixTmp = NULL;
    matrixTmp = pgmOpen((char *) "i.pgm", szx, szy);
    int matrix[szx][szy];
    for (int i = 0; i < szy; ++i) {
        for (int j = 0; j < szx; ++j) {
            matrix[i][j] = matrixTmp[i][j];
        }
    }
    /*define block size and length of zigzag matrix*/
    int LEN = BLOCK * BLOCK * PERCENT / 100;
    const int len = (LEN > BLOCK * BLOCK ? BLOCK * BLOCK : LEN);

    /**encoding part*/
    int *Encodearr = (int *) malloc(2 * len * sizeof(int));
    zigzagEncodeArray(Encodearr, len);

    /*zigzag index encode array*/
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

    /*define CUDA*/
    int blockX = (szx + BLOCK - 1) / BLOCK, blockY = (szy + BLOCK - 1) / BLOCK;
    dim3 block(BLOCK, BLOCK);
    dim3 grid(blockX, blockY);
    /*get output of zigzag array*/
    int *d_outarr = NULL;
    errCUDA(cudaMalloc((void **) &d_outarr, blockX * blockY * len * sizeof(int)));
    CUDA_MAIN<<<grid, block>>>(d_matrix, d_outarr, d_Qmatrix8, d_Encodearr, len, blockX, szx, szy);

    /*test: print zigzag array index*/
    int *outarr = (int *) malloc(blockX * blockY * len * sizeof(int));
    errCUDA(cudaMemcpy(outarr, d_outarr, blockX * blockY * len * sizeof(int), cudaMemcpyDeviceToHost));
	/*cout << "Zigzag array:" << endl;
	for (int i = 0; i < blockX * blockY * len; ++i) {
		cout << outarr[i] << " ";
	}
	cout << endl;*/

    /*output matrix*/
    int matrix2[szx][szy];
    errCUDA(cudaMemcpy(matrix2, d_matrix, szx * szy * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < szy; ++i) {
        for (int j = 0; j < szx; ++j) {
            matrixTmp[i][j] = saturation8bit(matrix2[i][j]);
        }
    }

    /*write image file*/
    pgmWrite((char *) "o.pgm", matrixTmp, szx, szy);

    /*destroy*/
    errCUDA(cudaFree(d_Encodearr));
    errCUDA(cudaFree(d_Qmatrix8));
    errCUDA(cudaFree(d_matrix));
    errCUDA(cudaFree(d_outarr));
    errCUDA(cudaDeviceReset());
    cout << "Hello, World!" << endl;

    return 0;
}
