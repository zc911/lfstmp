/**
 * convnet.h
 *
 * Author: dudalong (dudalong@baidu.com)
 * Created on: 2013-03-18
 *
 * Copyright (c) Baidu.com, Inc. All Rights Reserved
 *
 */
#ifndef CONVNET_H_
#define CONVNET_H_

#include <vector>
#include <string>
#include <time.h>
#include <math.h>

#include "layer.h"
#include "data.h"
#include "weights.h"
#include "util.h"
#include "fragment.h"

class Layer;

typedef struct {
    int slideWinSize;
    int winPoolSize;
    intv strideV;
}SlideWinType;

class ConvNet {
protected:
    SlideWinType _slideWinInfo;
    std::vector<Layer*> _layers;
    std::vector<Layer*> _outputLayerV;
    std::vector<Layer*> _dataLayerV;
    Layer *_outputLayer;
    
    virtual Layer* initLayer(string& layerType, dictParam_t &paramsDict,int isMultiPatch=0);
public:
    ConvNet(listDictParam_t &layerParams,int isMultiPatch=0);

    ~ConvNet();

    Layer* operator[](int idx);
    Layer* getLayer(int idx);

    int getNumLayers();
    SlideWinType &getSlideWinInfo();
    int cnnScore(Matrix &data, MatrixM &probsM);
    int cnnScore(Matrix &data, vector<string> &outlayer, MatrixM &featureM);
    int cnnScoreSlideWin(FragmentV &data, MatrixM &outValM, int strideX, int strideY);

    int initOutputMap(map<string, Matrix*> &outputMap);
    int releaseOutputMap(map<string, Matrix*> &outputMap);
    int setData(map<string, Matrix*> &outputMap, const vector<float*> &dataV, const vector<int> &imgWidthV,
            const vector<int> &imgHeightV, const vector<int> &imgChannelV, int dataNum);
    vector<int> getDataDimV();
    vector<int> getLabelsDimV();
    int cnnVarsizeImageScore(map<string, Matrix*> &outputMap, vector<float*> &outputV, vector<int> &outputLen);
    int cnn_varsize_image_score_caffe(map<string, Matrix*> &outputMap, 
                                      vector<float*> &outputV, vector<int> &outputLen);
};

#endif	/* CONVNET_H_ */

