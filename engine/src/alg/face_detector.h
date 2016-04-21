/*============================================================================
 * File Name   : face_detector.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/19/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_RANKER_ALG_FACE_DETECTOR_H_
#define MATRIX_RANKER_ALG_FACE_DETECTOR_H_

#include <queue>
#include <vector>

#include <glog/logging.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/render_face_detections.h>

using namespace cv;
using namespace std;

namespace dg 
{
struct BoundingBox {
    float confidence;
    Rect rect;
    Rect gt;
    bool deleted;
    int border;
    int id;
};

class FaceDetector
{
public:
    FaceDetector(string align_model, string avg_face);
    virtual ~FaceDetector();

    void Detect(vector<Mat>& images, vector<vector<Mat>>& vvResults, vector<vector<BoundingBox>>& vvBoxes);
    void Detect(Mat& image, vector<BoundingBox> boxes, vector<Mat>& results);
    void Align(vector<Mat>& images, vector<Mat>& results);

private:
    dlib::frontal_face_detector _detector;
    dlib::shape_predictor _sp;
    vector<dlib::point> _avg_face_points;

    bool predict(dlib::cv_image<dlib::bgr_pixel>& image, dlib::rectangle& bbox, vector<dlib::point>& points);
    Mat transform(dlib::cv_image<dlib::bgr_pixel>& image, vector<dlib::point>& points);
};

}

#endif //MATRIX_RANKER_ALG_FACE_DETECTOR_H_