//
// Created by jiajaichen on 16-9-27.
//

#ifndef PROJECT_FACE_ALIGNMENT_H
#define PROJECT_FACE_ALIGNMENT_H

#include <opencv2/opencv.hpp>

#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/render_face_detections.h>

using namespace std;
using namespace cv;
namespace dg {

class FaceAlignment {
public:
    typedef struct {

        bool is_model_encrypt = true;
        int batch_size = 1;
        string align_model;
        string align_deploy;
        vector<int> face_size;
    } FaceAlignmentConfig;
        std::vector<cv::Point>  landmarks;
    FaceAlignment(const FaceAlignmentConfig &config);

    void Align(std::vector<Mat> imgs,std::vector<Mat> &results,bool adjust=true);
private:
    void align_impl(const Mat &img, const Rect &bbox,std::vector<cv::Point> &landmarks);
    dlib::shape_predictor sp_;
    std::vector<cv::Point>  avg_points_;
    vector<int> face_size_;
};
}

#endif //PROJECT_FACE_ALIGNMENT_H
