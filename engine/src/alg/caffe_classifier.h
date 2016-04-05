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
#include <caffe/caffe.hpp>
#include "classifier.h"

namespace deepglint {

using namespace std;

typedef struct {
    int batch_size;
    int class_num;
    int target_min_size;
    int target_max_size;
    int rescale;
    int gpu_id;
    bool use_gpu;
    float means[3];
    string deploy_file;
    string model_file;
} CaffeConfig;

class CaffeClassifier : public Classifier {

 public:
    CaffeClassifier(CaffeConfig &caffeConfig) {
        device_setted_ = false;
        use_gpu_ = config.UseGPU;

        if (use_GPU_) {
            Caffe::SetDevice(config.gpu_id);
            gpu_id_ = config.gpu_id;
            Caffe::set_mode(Caffe::GPU);
            LOG(INFO) << "Use device " << config.gpu_id << endl;

        } else {
            LOG(WARNING) << "Use CPU only" << endl;
            Caffe::set_mode(Caffe::CPU);
        }
        batch_size_ = config.batch_size;

        net_.reset(new Net<float>(config.deploy_file, TEST));
        net_->CopyTrainedLayersFrom(config.model_file);
        CHECK_EQ(net_->num_inputs(), 1)
                << "Network should have exactly one input.";

        Blob<float>* input_layer = net_->input_blobs()[0];
        num_channels_ = input_layer->channels();
        CHECK(num_channels_ == 3 || num_channels_ == 1)
                << "Input layer should have 1 or 3 channels.";
        input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

        means_[0] = config.Means[0];
        means_[1] = config.Means[1];
        means_[2] = config.Means[2];

        target_min_size_ = config.TargetMinSize;
        target_max_size_ = config.TargetMaxSize;
        rescale_ = config.Rescale;
    }
    virtual ~CaffeClassifier() {

    }
    virtual std::vector<Blob<float>*> PredictBatch(vector<Mat> imgs,
                                                   float mean[3],
                                                   float rescale) = 0;
    virtual vector<vector<Prediction> > Classify(const vector<Mat> &imgs) = 0;
    virtual vector<vector<Prediction> > ClassifyBatch(
            const vector<Mat> &imgs) = 0;

 protected:

    int gpu_id_;
    int num_channels_;
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
