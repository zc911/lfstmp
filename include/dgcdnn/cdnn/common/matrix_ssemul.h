/**
 * matrix_ssemul.h
 *
 * Author: dudalong (dudalong@baidu.com)
 * Created on: 2013-04-11
 *
 * Copyright (c) Baidu.com, Inc. All Rights Reserved
 *
 */
#ifndef MATRIX_SSEMUL_H_
#define MATRIX_SSEMUL_H_
#include <xmmintrin.h> 

typedef union __attribute__ ((aligned (16))) {
        float f[4];
        __m128  v;
} __attribute__ ((aligned (16))) V4SF;

typedef unsigned short ushort;

typedef struct cscSparse_t
{
    ushort rows;
    ushort cols;
    ushort *rptr;
    int *cptr;
    int entryNum;
    float  *val;
} csc_t;


int SSEMatrixMul(float *A, float *TB, float *C, int m, int n, int k);

int mulBlock16SSE(float *a, float *b, float *c, int h, int w, int d);

int cDense2CscAlign16(ushort rows, ushort cols, float *cMat, csc_t *&cscMat);

int releaseCscMat(csc_t **cscMatPtr);

int rMatMulCscMatSSE8(float *rMat, csc_t *cscMat, float *resMat, ushort h, ushort w, ushort d);

int vecPairProduct16SSE(float *a, float *b, float *c, int n, int d, float scaleC=0);

#endif

