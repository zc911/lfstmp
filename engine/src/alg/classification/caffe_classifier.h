/*
 * caffe_classifier.h
 *
 *  Created on: 29/03/2016
 *      Author: chenzhen
 */

#ifndef CAFFE_CLASSIFIER_H_
#define CAFFE_CLASSIFIER_H_

#include <memory>
#include <opencv2/core/core.hpp>

#include "../caffe_config.h"
#include "classifier.h"
#include "caffe/caffe.hpp"

using namespace std;

namespace dg {

class CaffeClassifier : public Classifier {

 public:
    CaffeClassifier(CaffeConfig &caffeConfig)
            : gpu_id_(caffeConfig.gpu_id),
              batch_size_(caffeConfig.batch_size),
              target_min_size_(caffeConfig.target_min_size),
              target_max_size_(caffeConfig.target_max_size),
              use_gpu_(caffeConfig.use_gpu),
              device_setted_(false),
              meas_(caffeConfig.means) {

    }
    virtual ~CaffeClassifier() {

    }

    virtual vector<Prediction> Classify(const Mat &imgs) = 0;
    virtual vector<vector<Prediction> > ClassifyBatch(
            const vector<Mat> &imgs) = 0;

 protected:

    int gpu_id_;
    int batch_size_;
    float target_min_size_;
    float target_max_size_;
    bool use_gpu_;
    bool device_setted_;
    float means_[3];

    shared_ptr<caffe::Net<float> > net_;
    cv::Size input_geometry_;
};

}

#endif /* CAFFE_CLASSIFIER_H_ */
