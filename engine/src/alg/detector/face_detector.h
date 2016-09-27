#ifndef FACE_DETECTOR_H_INCLUDED
#define FACE_DETECTOR_H_INCLUDED

#include <string>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include "detector.h"
#include "model/model.h"

using namespace std;
using namespace cv;
using namespace caffe;

namespace dg {

class FaceCaffeDetector: public FaceDetector {
public:

    FaceCaffeDetector(const FaceDetectorConfig &config);

    virtual ~FaceCaffeDetector();
    int Detect(vector<cv::Mat> &img,
               vector<vector<Detection> > &detect_results);
private:
    void Forward( vector<Mat> &imgs, vector<vector<Detection> > &final_vbbox);
    void GetDetection(vector<Blob<float> *> &outputs,
                      vector<vector<Detection>> &final_vbbox, vector<cv::Mat> &imgs);
    void NMS(vector<Detection> &p, float threshold);

private:
    caffe::shared_ptr<Net<float> > net_;
    int num_channels_;
    int batch_size_;
    bool use_gpu_;
    vector<float> pixel_means_;
    float conf_thres_;
//    Size image_size_;
    unsigned int scale_;
    string layer_name_cls_;
    string layer_name_reg_;
    int img_scale_max_;
    int img_scale_min_;
    int sliding_window_stride_;
    vector<float> area_;
    vector<float> ratio_;
    vector<float> resize_ratios_;
    bool device_setted_ = false;
    int gpu_id_;
};

} /* namespace dg */

#endif /* FACE_DETECTOR_H_INCLUDED */
