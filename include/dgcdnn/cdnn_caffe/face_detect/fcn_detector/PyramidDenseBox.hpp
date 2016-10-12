/***************************************************************************
 * 
 * Copyright (c) 2016 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/



/**
 * @file face_detector/PyramidDenseBox.hpp
 * @author dukang01(com@baidu.com)
 * @date 2016/02/18 11:47:21
 * @brief 
 *  
 **/
#ifndef  __PYRAMID_DENSEBOX_H_
#define  __PYRAMID_DENSEBOX_H_
#include <string>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "fcn_detector.h"
#include "caffe_interface.h"
#include "util_others.hpp"
using namespace std;
using namespace cv;

using vis::ICaffeBlob;
using vis::ICaffePredict;
using vis::ICaffePreProcess;
using vis::createcaffepreprocess;
using vis::createcaffeblob;
using vis::CaffePreProcessParam;
using vis::CaffeParam;

class PyramidDenseBox{
    public:
        PyramidDenseBox()
        {
            templateSize_      = 24;
            minDetFaceSize_    = 24;
            maxDetFaceSize_    = -1 ;
            minImgSize_        = 20;
            maxImgSize_        = 2048;

            minScaleFaceToImg_ = 0.05f;
            maxScaleFaceToImg_ = 1.0f;
            stepScale_         = 2.0f;

            heat_map_a_        = 4.0f;
            heat_map_b_        = 2.0f;

            mean_r_            = 94.7109f;
            mean_g_            = 99.1183f;
            mean_b_            = 95.7652f;

            mvnPower_          = 1.0f;
            mvnScale_          = 0.01f;
            mvnShift_          = 0.0f;

            pad_w_             = 24;
            pad_h_             = 24;
            max_stride_        = 8;

            class_num_         = 1;
            channel_per_scale_ = 5;

            nms_threshold_     = 0.7f;
            nms_overlap_ratio_ = 0.5f;
            nms_top_n_         = 100;

        }
        PyramidDenseBox(float minDetFaceSize, float maxDetFaceSize)
        {
            templateSize_      = 24;
            minDetFaceSize_    = minDetFaceSize;
            maxDetFaceSize_    = maxDetFaceSize ;
            minImgSize_        = 20;
            maxImgSize_        = 2048;

            minScaleFaceToImg_ = 0.05f;
            maxScaleFaceToImg_ = 1.0f;
            stepScale_         = 2.0f;

            heat_map_a_        = 4.0f;
            heat_map_b_        = 2.0f;

            mean_r_            = 94.7109f;
            mean_g_            = 99.1183f;
            mean_b_            = 95.7652f;

            mvnPower_          = 1.0f;
            mvnScale_          = 0.01f;
            mvnShift_          = 0.0f;

            pad_w_             = 24;
            pad_h_             = 24;
            max_stride_        = 8;

            class_num_         = 1;
            channel_per_scale_ = 5;

            nms_threshold_     = 0.7f;
            nms_overlap_ratio_ = 0.5f;
            nms_top_n_         = 100;

        }
        ~PyramidDenseBox(){};

    private:
        int templateSize_;
        int minDetFaceSize_;
        int maxDetFaceSize_;
        int minImgSize_;
        int maxImgSize_;

        float minScaleFaceToImg_;
        float maxScaleFaceToImg_;
        float stepScale_;

        float heat_map_a_;
        float heat_map_b_;

        float mean_r_;
        float mean_g_;
        float mean_b_;

        float mvnPower_;
        float mvnScale_;
        float mvnShift_;

        int pad_w_;
        int pad_h_;
        int max_stride_;

        int class_num_;
        int channel_per_scale_;

        float nms_threshold_;
        float nms_overlap_ratio_;
        float nms_top_n_;

    public:
        //detection rbox
        bool predictPyramidDenseBox( ICaffePredict* caffe_net, Mat &img, vector< RotateBBox<float> >& rotatedFaces);
    private:
        bool setDetSize(const int imgWidth, const int imgHeight, const int template_size, 
                const int minDetFaceSize, const int maxDetFaceSize, 
                const int minImgSize, const int maxImgSize, 
                const float minScaleFaceToImg, const float maxScaleFaceToImg, 
                float& scale_start, float& scale_end);
        bool constructPyramidImgs(Mat & img, vector<Mat>& pyramidImgs); 
        template <typename Dtype> 
            void setImgDenseBox(Dtype* pblob, Mat& img, Scalar img_mean, float mvnPower, float mvnScale, float mvnShift);
        void predictDenseBox(ICaffePredict* caffe_net, Mat& img, vector< RotateBBox<float> >& faces);

};
#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
