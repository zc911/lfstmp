/*============================================================================
 * File Name   : image_service.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#include <vector>
#include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>
 
#include "image_service.h"
#include "codec/base64.h"
#include "io/uri_reader.h"

namespace dg 
{

::dg::MatrixError ImageService::ParseImage(const ::dg::Image& imgDes, ::cv::Mat& imgMat)
{
    if (imgDes.uri().size() > 0)
    {
        return getImageFromUri(imgDes.uri(), imgMat);
    }
    else if(imgDes.bindata().size() > 0)
    {
        return getImageFromData(imgDes.bindata(), imgMat);
    }

    ::dg::MatrixError err;
    err.set_code(-1);
    err.set_message("image URI or Data is required!");
    return err;
}

::dg::MatrixError ImageService::getImageFromData(const string img64, ::cv::Mat& imgMat)
{
    ::dg::MatrixError err;

    vector<uchar> bin;
    ::dg::Base64::Decode(img64, bin);
    if (bin.size() == 0)
    {
        err.set_code(-1);
        err.set_message("received empty image");
    }
    else
    {
        imgMat = ::cv::imdecode(::cv::Mat(bin), 1);
    }

    return err;
}

::dg::MatrixError ImageService::getImageFromUri(const string uri, ::cv::Mat& imgMat)
{
    ::dg::MatrixError err;

    vector<uchar> bin;
    int ret = UriReader::Read(uri, bin);
    if (ret < 0)
    {
        err.set_code(ret);
        err.set_message("load image failed!");
    }
    else
    {
        imgMat = ::cv::imdecode(::cv::Mat(bin), 1);
    }

    return err;
}

}
