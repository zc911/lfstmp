//
// Created by jiajaichen on 16-9-23.
//

#ifndef PROJECT_FACE_DLIB_DETECTOR_H
#define PROJECT_FACE_DLIB_DETECTOR_H

#include <stdexcept>
#include "dlib/image_processing.h"
#include "dlib/opencv.h"
#include "dlib/pixel.h"
#include "dlib/geometry.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "model/model.h"
#include "detector.h"

using namespace cv;
using namespace std;
namespace dg {


class FaceDlibDetector : public FaceDetector {

public:
    typedef struct {
        int img_scale_min = 240;
        int img_scale_max = 640;
    } FaceDetectorConfig;
    FaceDlibDetector(FaceDetectorConfig &config);
    FaceDlibDetector(int img_scale_min,int img_scale_max);
    virtual ~FaceDlibDetector();
    // detect only -> confidence, bbox
    int Detect(vector<cv::Mat> &img,
               vector<vector<Detection> > &detect_results);
    static void dlib_point2cv(const vector<dlib::point> &input, vector<cv::Point> &output) {
        output.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            const dlib::point &point = input[i];
            output[i] = Point(point.x(), point.y());
        }
    }

    static void dlib_rect2cv(const dlib::rectangle &input, Rect &output) {
        output = Rect(input.left(), input.top(), input.width(), input.height());
    }

    static void cv_point2dlib(const vector<cv::Point> &input, vector<dlib::point> &output) {
        output.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            const cv::Point &point = input[i];
            output[i] = dlib::point(point.x, point.y);
        }
    }

    static void cv_rect2dlib(const Rect &input, dlib::rectangle &output) {
        output = dlib::rectangle(input.x, input.y, input.x + input.width, input.y + input.height);
    }

private:
    dlib::frontal_face_detector _detector;
    int img_scale_max_;
    int img_scale_min_;
};
}
#endif //PROJECT_FACE_DLIB_DETECTOR_H
