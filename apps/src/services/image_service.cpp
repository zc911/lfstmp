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


MatrixError ImageService::ParseImage(vector<Image> &imgs,
                                     vector<cv::Mat> &imgMats,
                                     unsigned int timeout,
                                     bool concurrent) {
    MatrixError err;
    if (concurrent) {
        std::mutex pushmt, waitmt;
        std::condition_variable cv;
        for (int i = 0; i < imgs.size(); ++i) {
            pool->enqueue([&imgMats, &pushmt, &waitmt, &cv](Image &img, int size, unsigned int timeout) {
              cv::Mat mat;
              if (img.uri().size() > 0) {
                  getImageFromUri(img.uri(), mat, timeout);
              } else if (img.bindata().size() > 0) {
                  getImageFromData(img.bindata(), mat);
              } else {
                  // Bug here
                  this_thread::sleep_for(chrono::milliseconds(200));
              }

              std::unique_lock<mutex> pushlc(pushmt);
              imgMats.push_back(mat);
              pushlc.unlock();

              if (imgMats.size() == size) {
                  {
                      std::unique_lock<mutex> waitlc(waitmt);
                      cv.notify_all();
                  }
              }
            }, imgs[i], imgs.size(), timeout/2);

        }

        {
            std::unique_lock<mutex> waitlc(waitmt);
            if (cv.wait_for(waitlc, std::chrono::seconds(timeout)) == cv_status::no_timeout) {
                if (imgMats.size() != imgs.size()) {
                    LOG(ERROR) << "Parsed images size not equals to input size" << endl;
                    err.set_code(-1);
                    err.set_message("Parsed images size not equals to input size");
                }
            } else {
                LOG(ERROR) << "Parse input images timeout " << endl;
                err.set_code(-1);
                err.set_message("Parse input images timeout");
            }
        }
        return err;
    } else {
        for (int i = 0; i < imgs.size(); ++i) {
            Image img = imgs[i];
            cv::Mat mat;
            ParseImage(img, mat);
            imgMats.push_back(mat);
        }
        return err;
    }

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

MatrixError ImageService::getImageFromUri(const string uri, ::cv::Mat &imgMat, unsigned int timeout) {
    // whatever read, just return ok to let the batch proceed.
    MatrixError ok;
    vector<uchar> bin;
    int ret = UriReader::Read(uri, bin, timeout);
    if (ret == 0) {
        decodeDataToMat(bin, imgMat);
    }else{
        ok.set_code(-1);
        ok.set_message("Read image failed: " + uri);
        return ok;
    }

    if (imgMat.rows == 0 || imgMat.cols == 0) {
        LOG(ERROR) << "Image is empty: " << uri << endl;
    }
    return ok;

}

}
