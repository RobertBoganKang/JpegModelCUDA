#include <stdlib.h>
#include <cstdio>
#include <fstream>
#include <iostream>

#include "JpegDimension.cpp"

using namespace std;

int STOI(char *c) {
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
    imgx = STOI(str);
    fscanf(fp, "%s", str);
    imgy = STOI(str);
    fscanf(fp, "%s", str);
    int **imgdata = (int **) malloc(imgy * sizeof(int *));
    for (int i = 0; i < imgy; ++i) {
        imgdata[i] = (int *) malloc(imgx * sizeof(int));
    }
    for (int j = 0; j < imgy; ++j) {
        for (int i = 0; i < imgx; ++i) {
            fscanf(fp, "%s", str);
            imgdata[j][i] = STOI(str);
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
    for (int j = 0; j < imgy; ++j) {
        for (int i = 0; i < imgx; ++i) {
            myfile << imgdata[j][i] << "\t";
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
