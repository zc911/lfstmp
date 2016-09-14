//
// Created by jiajaichen on 16-9-5.
//

#ifndef SRC_ALG_CLASSIFICATION_BELT_CLASSIFIER_H_
#define SRC_ALG_CLASSIFICATION_BELT_CLASSIFIER_H_


#include <cassert>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/caffe.hpp"
#include "model/model.h"
#include "alg/detector/detector.h"
#include "alg/caffe_helper.h"
#include "alg/caffe_config.h"
#include <numeric>
using namespace std;
using namespace caffe;
using namespace cv;
namespace dg {
    typedef struct {

        bool is_model_encrypt = true;
        int batch_size = 1;
        int target_min_size = 400;
        int target_max_size = 1000;
        int gpu_id = 0;
        bool use_gpu = true;
        bool is_driver=true;
        string deploy_file;
        string model_file;
    } VehicleBeltConfig;
class CaffeBeltClassifier {
public:

    CaffeBeltClassifier(const VehicleBeltConfig &config);
    virtual ~CaffeBeltClassifier();
    vector<vector<Prediction> > ClassifyAutoBatch(const vector<Mat> &imgs);
protected:
    vector<vector<float> > ClassifyBatch(const vector<Mat> &imgs);

    std::vector<Blob<float> *> PredictBatch(vector<Mat> imgs);
    boost::shared_ptr<caffe::Net<float> > net_;
    int num_channels_;
    cv::Size input_geometry_;
    bool device_setted_;
    VehicleBeltConfig caffe_config_;
    Mat means_;
    int rescale_;
};

}
#endif //PROJECT_CAFFE_VEHICLE_COLOR_CLASSIFIER_H
