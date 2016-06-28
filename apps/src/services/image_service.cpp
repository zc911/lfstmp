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
#include "fs_util.h"
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

MatrixError ImageService::ParseImage(const WitnessImage &imgDes, ROIImages &imgroi) {
    Mat imgMat;
    MatrixError err;

    if (imgDes.data().uri().size() > 0) {
        err = getImageFromUri(imgDes.data().uri(), imgMat);
    } else if (imgDes.data().bindata().size() > 0) {
        err = getImageFromData(imgDes.data().bindata(), imgMat);
    } else {
        err.set_code(-1);
        err.set_message("image URI or Data is required!");
        return err;

    }
    std::vector<cv::Rect> rois;
    if (imgDes.relativeroi().size() > 0) {
        err = getRelativeROIs(imgDes.relativeroi(), rois);
    } else {
        err = getMarginROIs(imgDes.marginroi(), rois, imgMat);
    }
    imgroi.data = imgMat;
    imgroi.rois = rois;
    return err;
}

MatrixError ImageService::ParseImage(std::vector<WitnessImage> &imgs,
                                     std::vector<ROIImages> &roiimages,
                                     unsigned int timeout, bool concurrent) {
    MatrixError err;
    if (concurrent) {
        std::mutex countmt, waitmt;
        std::condition_variable cv;
        int finishCount = 0;
        roiimages.resize(imgs.size());
        for (int i = 0; i < imgs.size(); ++i) {
            pool->enqueue(
                [&roiimages, &finishCount, &countmt, &cv](WitnessImage &img,
                                                                   int size,
                                                                   unsigned int timeout,
                                                                   int index) {
                  cv::Mat mat;
                  if (img.data().uri().size() > 0) {
                      getImageFromUri(img.data().uri(), mat, timeout);
                  } else if (img.data().bindata().size() > 0) {
                      getImageFromData(img.data().bindata(), mat);
                  }

                  std::vector<cv::Rect> rois;
                  if (img.relativeroi().size() > 0) {
                      getRelativeROIs(img.relativeroi(), rois);
                  } else {
                      getMarginROIs(img.marginroi(), rois, mat);
                  }

                  ROIImages roiimage;
                  roiimage.data = mat;
                  roiimages[index] = roiimage;
                  {
                      std::unique_lock<mutex> countlc(countmt);
                      ++finishCount;
                  }
                  countmt.unlock();
                  if (finishCount == size) {
                      {
//                          std::unique_lock<mutex> waitlc(waitmt);
                          cv.notify_all();
                      }
                  }
                },
                imgs[i], imgs.size(), timeout / 2, i);

        }

        {
            std::unique_lock<mutex> waitlc(waitmt);
            if (cv.wait_for(waitlc,
                            std::chrono::seconds(timeout),
                            [&finishCount, &imgs]() { return finishCount == imgs.size(); })) {
                if (roiimages.size() != imgs.size()) {
                    LOG(ERROR) << "Parsed images size not equals to input size" << std::endl;
                    err.set_code(-1);
                    err.set_message("Parsed images size not equals to input size");
                }

            } else {
                LOG(ERROR) << "Parse input images timeout " << std::endl;
                err.set_code(-1);
                err.set_message("Parse input images timeout");
            }

        }
        return err;

    } else {

        for (int i = 0; i < imgs.size(); ++i) {
            WitnessImage img = imgs[i];
            ROIImages roiimage;
            ParseImage(img, roiimage);
            roiimages.push_back(roiimage);
        }
        return err;
    }

}
MatrixError ImageService::getRelativeROIs(
    ::google::protobuf::RepeatedPtrField<::dg::model::WitnessRelativeROI> roisSrc,
    std::vector<cv::Rect> &rois) {
    MatrixError ok;
    for (::google::protobuf::RepeatedPtrField<
        ::dg::model::WitnessRelativeROI>::iterator it =
        roisSrc.begin(); it != roisSrc.end(); it++) {
        cv::Rect rect(it->posx(), it->posy(), std::max(it->width(), 0),
                      std::max(0, it->height()));
        rois.push_back(rect);
    }

    return ok;
}
MatrixError ImageService::getMarginROIs(
    ::google::protobuf::RepeatedPtrField<::dg::model::WitnessMarginROI> roisSrc,
    std::vector<cv::Rect> &rois, const cv::Mat &img) {
    MatrixError ok;
    int width = img.cols;
    int height = img.rows;
    for (::google::protobuf::RepeatedPtrField<
        ::dg::model::WitnessMarginROI>::iterator it = roisSrc.begin();
         it != roisSrc.end(); it++) {
        Margin margin;
        margin.left = it->left();
        margin.top = it->top();
        margin.right = it->right();
        margin.bottom = it->bottom();

        cv::Rect rect(0 + margin.left, 0 + margin.top,
                      width - margin.right - margin.left,
                      height - margin.top - margin.top);

        rois.push_back(rect);
    }

    return ok;

}
static void decodeDataToMat(vector<uchar> &data, cv::Mat &imgMat) {
    if (data.size() >= 0) {
        try {
            imgMat = ::cv::imdecode(::cv::Mat(data), 1);
        } catch (exception &e) {
            LOG(WARNING) << "decode image failed: " << e.what() << endl;
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
        LOG(WARNING) << "Image is empty from BASE64" << endl;
    }
    return ok;
}

MatrixError ImageService::getImageFromUri(const string uri, ::cv::Mat &imgMat,
                                          unsigned int timeout) {
    // whatever read, just return ok to let the batch proceed.
    MatrixError ok;
    vector<uchar> bin;
    int ret = UriReader::Read(uri, bin, timeout);
    string test=Base64::Encode(bin);
    WriteToFile("warmup.dat",(char *)test.c_str(),test.length());
    if (ret == 0) {
        decodeDataToMat(bin, imgMat);
    } else {
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
