/*
 * caffe_detector.h
 *
 *  Created on: Sep 28, 2015
 *      Author: irene
 */

#ifndef SRC_ALG_CAFFE_DETECTOR_H_
#define SRC_ALG_CAFFE_DETECTOR_H_

#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ctime>
#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/caffe.hpp"
#include "model/basic.h"
#include "detector.h"
#include "caffe_config.h"

using namespace std;

namespace dg {

class CaffeDetector : public Detector {
 public:
    CaffeDetector(const CaffeConfig &config) {
        model_file_ = config.model_file;
        deploy_file_ = config.deploy_file;
        target_min_size_ = config.target_min_size;
        target_max_size_ = config.target_max_size;

        use_gpu_ = config.use_gpu;
        gpu_id_ = config.gpu_id;
        rescale_ = config.rescale;

        batch_size_ = config.batch_size;
        num_channels_ = 0;

    }
    virtual ~CaffeDetector() {

    }

    virtual vector<Detection> Detect(const cv::Mat &img) = 0;
    virtual vector<vector<Detection>> DetectBatch(
            const vector<cv::Mat> &img) = 0;

 protected:
    boost::shared_ptr<caffe::Net<float> > net_;
    string model_file_;
    string deploy_file_;

    float target_min_size_;
    float target_max_size_;
    int batch_size_;

    bool use_gpu_;
    int gpu_id_;
    bool rescale_;
    int num_channels_;

};

}
#endif /* SRC_ALG_CAFFE_DETECTOR_H_ */
