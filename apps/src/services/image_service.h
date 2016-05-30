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

#include "../model/common.pb.h"
#include "simple_thread_pool.h"

namespace dg {

using namespace ::dg::model;

const int IMAGE_SERVICE_THREAD_NUM = 8;

class ImageService {
public:
    static MatrixError ParseImage(const ::dg::model::Image &image, ::cv::Mat &imgMat);
    static MatrixError
        ParseImage(vector<Image> &imgs, vector<cv::Mat> &imgMats, unsigned int timeout, bool concurrent = true);

private:
    static MatrixError getImageFromUri(const std::string uri,
                                       ::cv::Mat &imgMat, unsigned int timeout = 10);
    static MatrixError getImageFromData(const std::string img64,
                                        ::cv::Mat &imgMat);

    static ThreadPool *pool;
};

}

#endif //MATRIX_APPS_IMAGE_SERVICE_H_
