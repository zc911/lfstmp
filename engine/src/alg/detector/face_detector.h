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
    typedef struct {
        bool is_model_encrypt = false;
        int batch_size = 1;
        int gpu_id = 0;
        int img_scale_max = 300;
        int img_scale_min = 240;
        float scale = 1.0f;
        float confidence = 0.7;
        bool use_gpu = true;
        string deploy_file;
        string model_file;
        string layer_name_cls = "conv_face_16_cls";
        string layer_name_reg = "conv_face_16_reg";
    } FaceDetectorConfig;
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
