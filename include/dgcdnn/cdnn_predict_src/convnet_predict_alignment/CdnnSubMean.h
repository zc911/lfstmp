/***************************************************************************
 * 
 * Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file cdnn_para.h
 * @author vis(com@baidu.com)
 * @date 2014/12/30 13:16:05
 * @brief 
 *  
 **/

#ifndef  __CDNN_PARA_H_
#define  __CDNN_PARA_H_

#include "cdnn_score.h"
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

typedef struct dcnn_predict_params
{
    float* mean_data;
    int labels_dim;
    int data_dim;
    int img_size;
}dcnn_predict_params;

// private
int init_dcnn_model(const char* model_path, const char* mean_path, int &feature_length, void* &initModel, dcnn_predict_params& dcnn_params);
int release_dcnn_model(void *&initModel, dcnn_predict_params dcnn_params);
int get_dcnn_score_ipl(void* image, float* probs, void *& initModel, dcnn_predict_params dcnn_params, int cropBorder);
int get_input_data_for_dcnn_ipl(IplImage* src_img, float* data, float* mean_data, int size, int cropBorder);

#endif  //__CDNN_PARA_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
