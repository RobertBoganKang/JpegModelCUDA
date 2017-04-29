#ifndef SOLUTION_JPEGCOMMON_H
#define SOLUTION_JPEGCOMMON_H

#include <iostream>
#include <fstream>
#include "JpegDimension.cpp"

using namespace std;
int STOI(char *c);

int **pgmOpen(char *file, int &imgx, int &imgy);

void pgmWrite(char *file, int **imgdata, int imgx, int imgy);

extern int Qmatrix8[BLOCK][BLOCK];

int saturation8bit(int i);

void zigzagEncodeArray(int *array, int len);

#endif //SOLUTION_JPEGCOMMON_H
