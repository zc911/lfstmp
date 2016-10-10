/**
 * conv_util.h
 *
 * Author: dudalong (dudalong@baidu.com)
 * Created on: 2013-03-30
 *
 * Copyright (c) Baidu.com, Inc. All Rights Reserved
 *
 */
#ifndef CONV_UTIL_H_
#define	CONV_UTIL_H_

#include "fragment.h"
#include "matrix.h"
#include "matrix_ssemul.h"
#include "util.h"
#define BLAS2SSE_THRESH 16

void convFilterActsUnroll(Matrix& images, Matrix& filters, Matrix& targetss, int *offsetIn, int *offsetOut,
        int imgSizeX, int numModulesX, int paddingStart, int moduleStride,
        int numImgColors, int groups, float scaletarget, float scaleOutput);

void deconvFilterActsUnroll(Matrix& images, Matrix& filters, Matrix& targetss, int *offsetIn, int *offsetOut,
        int imgSizeX, int numModulesX, int paddingStart, int moduleStride,
        int numImgColors, int groups, float scaletarget, float scaleOutput);

void convFilterActsUnrollFragment(FragmentV& fragments, Matrix& filters, FragmentV& targets,
        int moduleStride, int numImgColors, int groups, float scaleTargets, float scaleOutput);

void convFilterActsUnrollVarsize(Matrix& input, Matrix& filters, Matrix& output,
        int moduleStride, int numImgColors, int groups, int padding);

void conv_filter_acts_unroll_varsize_caffe(Matrix& input, Matrix& filters,
                                           Matrix& output, int moduleStride, int numImgColors,
                                           int groups, int padding);

void convLocalPoolMax(Matrix& images, Matrix& targets, int numFilters,
                   int subsX, int startX, int strideX, int outputsX);

void convLocalPoolMaxFragment(FragmentV& fragments, FragmentV& targets, int subsX, int strideX, int winOutputsX);

void convLocalPoolMaxVarsize(Matrix& inputs, Matrix& outputs, int channels,
         int subsX, int strideX, int strideY, int startX, int startY);

void conv_local_poolmax_varsize_caffe(Matrix& inputs, Matrix& outputs,
                                      int channels, int subsX, int strideX,
                                      int strideY, int startX, int startY);

void convLocalPoolAvg(Matrix& images, Matrix& targets, int numFilters,
                   int subsX, int startX, int strideX, int outputsX);

void conv_local_pool_avg_caffe(Matrix& images, Matrix& targets, int numFilters,
                   int subsX, int startX, int strideX, int outputsX);

void conv_local_roipool(Matrix& images, Matrix& targets, Matrix *& _tmpMat, 
                        int inpIdx, int numFilters, int poolW, int poolH,
                        float spatialScale, int startX, int outputsX);

void conv_local_roipool_varsize(Matrix& images, Matrix& targets, Matrix *& _tmpMat,
                             int numFilters, int poolW, int poolH,
                             float spatialScale, int startX, int outputsX);

void convContrastNorm(Matrix& images, Matrix& meanDiffs, 
        Matrix& targets, int numFilters, int sizeX, float addScale, float powScale);

void convResponseNorm(Matrix& images, Matrix& targets, int numFilters, int sizeX, float addScale, float powScale);

void convContrastNormCrossMap(Matrix& images, Matrix& meanDiffs, 
                    Matrix& targets, int numFilters, int sizeF, float addScale, float powScale);

void convResponseNormCrossMap(Matrix& images, Matrix& targets, int numFilters, int sizeF, float addScale, float powScale);

void convContrastNormCrossMapVarsize(Matrix& images, Matrix& meanDiffs, 
                    Matrix& targets, int numFilters, int sizeF, float addScale, float powScale);

void conv_contrast_norm_crossmap_varsize_caffe(Matrix& images, Matrix& meanDiffs, 
                                               Matrix& targets, int numFilters, int sizeF,
                                               float addScale, float powScale, float kconst);

void convResponseNormCrossMapVarsize(Matrix& images, Matrix& targets, int numFilters, int sizeF, float addScale, float powScale);

void convContrastNormCrossMapFragment(FragmentV &fragments, FragmentV &meanDiffs, FragmentV& targets, int numFilters, 
        int sizeF, float addScale, float powScale);

void convResponseNormCrossMapFragment(FragmentV &fragments, FragmentV& targets, int numFilters,
        int sizeF, float addScale, float powScale);

void convAddBiases(Matrix& biases, Matrix& targets, int numModules, bool sharedBiases);

void deconvAddBiases(Matrix& biases, Matrix& targets, int numModules, bool sharedBiases);

void convAddSharedBiasesFragment(Matrix& biases, FragmentV& targets, int numFilters);

void arrangeSlideWinConvOutput(FragmentV& fragments, int inputWidth, int inputHeight,
        int strideX, int strideY, int winSize, int winPoolSize, intv &strideV, Matrix& targets);

void depth_concat(Matrix& input, int inpIdx, float scaleTargets, intv *depth, Matrix& targets);

void fcWeightMul(Matrix& input, Matrix& weight, float scaleTargets, float scaleOutput, Matrix& targets);

void fcWeightMulSparse(Matrix& input, csc_t* weight, float scaleTargets, float scaleOutput, Matrix& targets);

void fcAddBiases(Matrix& biases, Matrix& targets);

void localFilterActsUnroll(Matrix& images, Matrix& filters, Matrix& targetss, int *offsetIn, int *offsetOut,
        int imgSizeX, int numModulesX, int paddingStart, int moduleStride,
        int numImgColors, float scaletarget, float scaleOutput);

void localAddBiases(Matrix& biases, Matrix& targets, int numModules);

void softmax(Matrix& inputs, Matrix& outputs);

void hybridReluTanh(Matrix& inputs, Matrix& outputs);//by xiatian

void blockExpand(Matrix &inputs, Matrix &outputs, int channels, int blockSizeX,
        int blockSizeY, int blockOutputX, int blockOutputY, int strideX, int strideY, int paddingX, int paddingY);
int getNewState(Matrix &state, Matrix &fg, Matrix &inputUnit, Matrix &ig, int blockNum, int cellsPerBlock);

void maxout(Matrix& inputs, Matrix& outputs, int channels, int groups);

#endif	/* CONV_UTIL_H_ */

