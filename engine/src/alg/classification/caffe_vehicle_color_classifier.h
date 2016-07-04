//
// Created by jiajaichen on 16-6-27.
//

#ifndef PROJECT_CAFFE_VEHICLE_COLOR_CLASSIFIER_H
#define PROJECT_CAFFE_VEHICLE_COLOR_CLASSIFIER_H



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

class  CaffeVehicleColorClassifier{
public:
    typedef struct {

        bool is_model_encrypt = false;
        int batch_size = 1;
        int target_min_size = 400;
        int target_max_size = 1000;
        int gpu_id = 0;
        bool use_gpu = true;
        string deploy_file;
        string model_file;
    } VehicleColorConfig;
    CaffeVehicleColorClassifier(const VehicleColorConfig &config);
    virtual ~CaffeVehicleColorClassifier();
    vector<vector<Prediction> > ClassifyAutoBatch(const vector<Mat> &imgs);
protected:
    vector<vector<Prediction> > ClassifyBatch(const vector<Mat> &imgs);

    std::vector<Blob<float>*> PredictBatch(vector<Mat> imgs);
    void WrapBatchInputLayer(vector<vector<Mat> > *input_batch);

    void PreprocessBatch(const vector<Mat> imgs,
                         vector<vector<Mat> >* input_batch);

    boost::shared_ptr<caffe::Net<float> > net_;
    int num_channels_;
    cv::Size input_geometry_;
    bool device_setted_;
    VehicleColorConfig caffe_config_;
    Mat means_;
    int rescale_;
};

}
#endif //PROJECT_CAFFE_VEHICLE_COLOR_CLASSIFIER_H
