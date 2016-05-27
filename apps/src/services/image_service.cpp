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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "image_service.h"
#include "codec/base64.h"
#include "io/uri_reader.h"

namespace dg {

ThreadPool *ImageService::pool = new ThreadPool(8);

MatrixError ImageService::ParseImage(const Image &imgDes, ::cv::Mat &imgMat) {
    if (imgDes.uri().size() > 0) {
        return getImageFromUri(imgDes.uri(), imgMat);
    } else if (imgDes.bindata().size() > 0) {
        return getImageFromData(imgDes.bindata(), imgMat);
    }

    MatrixError err;
    err.set_code(-1);
    err.set_message("image URI or Data is required!");
    return err;
}

MatrixError ImageService::ParseImage(vector<Image> &imgs, vector<cv::Mat> &imgMats, bool concurrent) {
    if (concurrent) {
        std::mutex lock;
        for (int i = 0; i < imgs.size(); ++i) {

            pool->enqueue([&imgMats, &lock](Image &img) {
              cv::Mat mat;
              if (img.uri().size() > 0) {
                  getImageFromUri(img.uri(), mat);
              } else if (img.bindata().size() > 0) {
                  getImageFromData(img.bindata(), mat);
              }
              lock.lock();
              imgMats.push_back(mat);
              lock.unlock();
            }, imgs[i]);

        }
        while (imgMats.size() < imgs.size()) {
            cout << imgMats.size() << ":" << imgs.size() << endl;
        }
        cout << "Finish read" << endl;
        cout << "Read finish:" << imgMats.size() << endl;

    } else {
        for (int i = 0; i < imgs.size(); ++i) {
            Image img = imgs[i];
            cv::Mat mat;
            ParseImage(img, mat);
            imgMats.push_back(mat);
        }
    }
    MatrixError err;
    err.set_code(-1);
    err.set_message("image URI or Data is required!");
    return err;
}

static void decodeDataToMat(vector<uchar> &data, cv::Mat &imgMat) {
    if (data.size() >= 0) {
        try {
            imgMat = ::cv::imdecode(::cv::Mat(data), 1);
        }
        catch (exception &e) {
            LOG(ERROR) << "decode image failed: " << e.what() << endl;
        }
    }
}

MatrixError ImageService::getImageFromData(const string img64,
                                           ::cv::Mat &imgMat) {
    MatrixError ok;
    vector<uchar> bin;
    Base64::Decode(img64, bin);
    decodeDataToMat(bin, imgMat);

    if ((imgMat.rows & imgMat.cols) == 0) {
        LOG(ERROR) << "Image is empty from BASE64" << endl;
    }
    return ok;
}

MatrixError ImageService::getImageFromUri(const string uri, ::cv::Mat &imgMat) {
    // whatever read, just return ok to let the batch proceed.
    MatrixError ok;
    vector<uchar> bin;
    int ret = UriReader::Read(uri, bin);
    if (ret == 0) {
        decodeDataToMat(bin, imgMat);
    }

    if (imgMat.rows == 0 || imgMat.cols == 0) {
        LOG(ERROR) << "Image is empty: " << uri << endl;
    }
    return ok;

}

}
