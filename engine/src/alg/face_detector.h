#ifndef FACE_DETECTOR_H_INCLUDED
#define FACE_DETECTOR_H_INCLUDED

#include <string>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include "model/basic.h"
#include "model/model.h"
#include "caffe_helper.h"

using namespace std;
using namespace cv;
using namespace caffe;

namespace dg {

class FaceDetector {
 public:
    FaceDetector(const string& model_file, const string& trained_file,
                 const bool use_gpu, const int batch_size,
                 const Size &image_size, const float conf_thres);

    virtual ~FaceDetector();
    vector<vector<Detection>> Detect(vector<Mat> imgs);

 private:
    void Forward(const vector<Mat> &imgs, vector<Blob<float>*> &outputs);
    void GetDetection(vector<Blob<float>*>& outputs,
                      vector<vector<Detection> > &final_vbbox);

 private:
    caffe::shared_ptr<Net<float> > net_;
    int num_channels_;
    int batch_size_;
    bool use_gpu_;
    vector<float> pixel_means_;
    float conf_thres_;
    Size image_size_;
    string layer_name_cls_;
    string layer_name_reg_;
    int sliding_window_stride_;
    vector<float> area_;
    vector<float> ratio_;
};

} /* namespace dg */

#endif /* FACE_DETECTOR_H_INCLUDED */

