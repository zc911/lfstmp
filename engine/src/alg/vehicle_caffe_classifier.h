/*
 * vehicle_caffe_classifier.h
 *
 *  Created on: Apr 21, 2016
 *      Author: jiajaichen
 */

#ifndef SRC_ALG_VEHICLE_CAFFE_CLASSIFIER_H_
#define SRC_ALG_VEHICLE_CAFFE_CLASSIFIER_H_
#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <caffe/caffe.hpp>
#include "model/model.h"
#include "detector.h"
#include "caffe_helper.h"
#include "caffe_config.h"
using namespace std;
using namespace caffe;
using namespace cv;
namespace dg {

class VehicleCaffeClassifier {
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
    } VehicleCaffeConfig;
    VehicleCaffeClassifier(const VehicleCaffeConfig &config);
    virtual ~VehicleCaffeClassifier();
    vector<vector<Prediction> > ClassifyAutoBatch(const vector<Mat> &imgs);
 protected:
    vector<vector<Prediction> > ClassifyBatch(const vector<Mat> &imgs);

    std::vector<Blob<float>*> PredictBatch(vector<Mat> imgs);
    void WrapBatchInputLayer(vector<vector<Mat> > *input_batch);

    void PreprocessBatch(const vector<Mat> imgs,
                         vector<vector<Mat> >* input_batch);
 private:
    boost::shared_ptr<caffe::Net<float> > net_;
    int num_channels_;
    cv::Size input_geometry_;
    bool device_setted_;
    VehicleCaffeConfig caffe_config_;
    Mat means_;
    int rescale_;
};

} /* namespace dg */

#endif /* SRC_ALG_VEHICLE_CAFFE_CLASSIFIER_H_ */
