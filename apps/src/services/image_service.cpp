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
 
#include "image_service.h"
#include "codec/base64.h"

::dg::MatrixError ImageService::ParseImage(const ::dg::Image *imgDes, ::cv::Mat& imgMat)
{
    ::dg::MatrixError err;
    if (imgDes == NULL || imgDes->uri_.size() == 0 || imgDes->bindata_.size() == 0)
    {
        err.set_code(-1);
        err.set_message("image URI or Data is required!");
        return err;
    }

    vector<uchar> imgBin;
    if (imgDes->uri_.size() > 0)
    {
        imgBin = getImageFromUri(imgDes->uri_);
    }
    else 
    {
        imgBin = getImageFromData(imgDes->bindata_);
    }

    if (imgBin.size() == 0)
    {
        err.set_code(-1);
        err.set_message("received empty image");
        return err;
    }

    imgMat = imdecode(Mat(imgBin), 1);
    return err;
}

vector<uchar> ImageService::getImageFromData(const string img64)
{
    vector<uchar> bin;
    Base64::Decode(img64, bin);
    return bin;
}

vector<uchar> ImageService::getImageFromUri(const string uri)
{
    return vector<uchar>();
}