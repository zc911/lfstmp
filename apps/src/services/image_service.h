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

namespace dg {

using namespace ::dg::model;

class ImageService {
 public:
    static MatrixError ParseImage(const Image& image, ::cv::Mat& imgMat);

 private:
    static MatrixError getImageFromUri(const std::string uri,
                                       ::cv::Mat& imgMat);
    static MatrixError getImageFromData(const std::string img64,
                                        ::cv::Mat& imgMat);
};

}

#endif //MATRIX_APPS_IMAGE_SERVICE_H_
