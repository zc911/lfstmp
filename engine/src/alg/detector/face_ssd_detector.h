//
// Created by jiajaichen on 16-9-27.
//

#ifndef PROJECT_FACE_SSD_DETECTOR_H
#define PROJECT_FACE_SSD_DETECTOR_H

#include <string>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include "detector.h"
#include "model/model.h"

using namespace std;
using namespace cv;
using namespace caffe;
namespace dg {
class FaceSsdDetector: public FaceDetector {
public:
    FaceSsdDetector(const FaceDetectorConfig &config);
    virtual ~FaceSsdDetector();
    int Detect(vector<cv::Mat> &img,
               vector<vector<Detection> > &detect_results);

protected:
    vector<Blob<float>*> PredictBatch( vector<Mat> &imgs);
    void Fullfil(vector<cv::Mat> &img, vector<Blob<float> *> &outputs, vector<vector<Detection> > &detect_results);
    boost::shared_ptr<caffe::Net<float>> net_;
    int num_channels_;
    cv::Size input_geometry_;
    bool device_setted_;
    VehicleCaffeDetectorConfig caffe_config_;
    bool use_gpu_;
    int gpu_id_;
    int batch_size_;
    float threshold_;
    int target_col_;
    int target_row_;
    int max_row_;
    int max_col_;
    vector<float> resized_ratio_;

};
}
#endif //PROJECT_FACE_SSD_DETECTOR_H
