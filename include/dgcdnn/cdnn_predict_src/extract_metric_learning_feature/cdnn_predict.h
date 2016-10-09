/***************************************************************************
 * 
 * Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
/**
 * @file cdnn_predict.h
 * @author vis(com@baidu.com)
 * @date 2015/05/27 17:44:37
 * @brief 
 *  
 **/

#ifndef  __CDNN_PREDICT_H_
#define  __CDNN_PREDICT_H_

#include <vector>
#include <string>

IplImage* cvGetSubImage(IplImage *image, CvRect roi,int dst_width, int dst_height);

int getPatch(IplImage* img,float x, float y,int scale,int dstsize,float* patchdata);

int prepdata(IplImage* src, int dstsize,int cropborder, std::vector<float>& pts, std::vector<int>& scale, int patchsize,float* data_mean_src,float** patchdata,int& patchlen);

int cdnn_predict(IplImage* img, int dstsize, int cropborder, std::vector<float>& pts, std::vector<int>& scale, int patchsize, float* data_mean, void *model, float* probs);

int cdnn_extract_feature(IplImage* img, int dstsize, int cropborder, std::vector<float>& pts, std::vector<int>& scale, int patchsize, float* data_mean, void *model, std::vector<std::string> &outlayer, float*& probs, int& outFeatDim);

#endif  //__CDNN_PREDICT_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
