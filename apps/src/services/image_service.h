/*============================================================================
 * File Name   : image_service.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_IMAGE_SERVICE_H_
#define MATRIX_APPS_IMAGE_SERVICE_H_

#include <opencv2/core/core.hpp>

#include "common.pb.h"
#include "witness.pb.h"
#include "matrix_engine/model/frame.h"

#include "simple_thread_pool.h"

namespace dg {
typedef struct {
    cv::Mat data;
    std::vector<cv::Rect> rois;
} ROIImages;
using namespace ::dg::model;

const int IMAGE_SERVICE_THREAD_NUM = 8;

class ImageService {
 public:
    static MatrixError ParseImage(const WitnessImage &imgDes,
                                  ROIImages &imgMat);
    MatrixError static ParseImage(vector<WitnessImage> &imgs,
                                  vector<ROIImages> &imgMats,
                                  unsigned int timeout, bool concurrent = true);
    static MatrixError ParseImage(const Image &imgDes, ::cv::Mat &imgMat);

 private:
    static MatrixError getImageFromUri(const std::string uri, ::cv::Mat &imgMat,
                                       unsigned int timeout = 10);
    static MatrixError getImageFromData(const std::string img64,
                                        ::cv::Mat &imgMat);
    static MatrixError getRelativeROIs(
            ::google::protobuf::RepeatedPtrField<::dg::model::WitnessRelativeROI>,
            std::vector<cv::Rect> &rois);
    static MatrixError getMarginROIs(
            ::google::protobuf::RepeatedPtrField<::dg::model::WitnessMarginROI>,
            std::vector<cv::Rect> &rois, const cv::Mat &img);
    static ThreadPool *pool;
};

}

#endif //MATRIX_APPS_IMAGE_SERVICE_H_
