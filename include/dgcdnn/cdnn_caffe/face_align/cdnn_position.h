/***************************************************************************
 * 
 * Copyright (c) 2013 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
/**
 * @file ../include/cdnn_predict.h
 * @author xiatian(com@baidu.com)
 * @date 2013/04/19 18:17:42
 * @brief 
 *  
 **/
#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <cv.h>
#include <highgui.h>
using namespace std;
using namespace cv;
using namespace std;

typedef struct dcnn_predict_params
{
    float * mean_data;
    int labels_dim;
    int data_dim;
    int img_size;
}dcnn_predict_params;

typedef struct KEYPOINTS_MODEL
{
    void* rectModel;
    dcnn_predict_params rectCdnnParams;
    void* fineRectModel;
    dcnn_predict_params fineRectCdnnParams;
    void* ptsModel;
    dcnn_predict_params ptsCdnnParams;
}
sKeyPointsModel;

