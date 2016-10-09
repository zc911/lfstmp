/**
 * cdnn_score.h
 *
 * Author: dudalong (dudalong@baidu.com)
 * Created on: 2013-03-16
 *
 * Copyright (c) Baidu.com, Inc. All Rights Reserved
 *
 */
#ifndef CDNNSCORE_H_
#define	CDNNSCORE_H_

#include <vector>
#include <string>
using namespace std;

int cdnnInitModel(const char *filePath, void *&model,int isMultipatch=0, bool useBlas=true, bool useMultiThreadBlas=false);

int cdnnScore(float *data, void *model, int dataNum, int dataDim, float *probs);

int cdnn_score_caffe(float *data, void *model, int dataNum, int dataDim, float *probs, 
                                                                        int chns = 3);

int cdnnScoreMultiPatch(float *data, void *model, int dataNum, int dataDim, float *probs);


int cdnnFeatExtract(float *data, void *model, int dataNum, int dataDim,
        vector<string> &outlayer, float *&outFeat, int &outFeatDim);

int cdnnGetSlideWinSize(void *model);

int cdnnGetSlideWinNum(void *model, int imgHeight, int imgWidth, int strideX, int strideY);

int cdnnScoreSlideWin(float *data, void *model, int imgWidth, int imgHeight,
        int strideX, int strideY, int imgChannel, float *outVal);

int cdnnGetDataDimV(void *model, vector<int> &dataDimV);

int cdnnVarsizeImageScore(const vector<float *> &dataV, void *model, const vector<int> &imgWidthV,
        const vector<int> &imgHeightV, const vector<int> &imgChannelV, int dataNum, vector<float *> &outValV,
        vector<int> &outLenV);

int cdnn_varsize_image_score_caffe(const vector<float *> &dataV, void *model, 
                                   const vector<int> &imgWidthV, const vector<int> &imgHeightV,
                                   const vector<int> &imgChannelV, int dataNum,
                                   vector<float *> &outValV, vector<int> &outLenV);

int cdnnReleaseModel(void **model);

int cdnnGetDataDim(void *model);

int cdnnGetLabelsDim(void *model);

int cdnn_get_data_channel(void* model);

#endif	/* CDNNSCORE_H_ */

