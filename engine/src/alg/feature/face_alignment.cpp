//
// Created by jiajaichen on 16-9-27.
//
#include "face_alignment.h"
#include "../detector/face_dlib_detector.h"
namespace dg {
void FaceAlignment::Align(std::vector<Mat> imgs, std::vector<Mat> &results, bool adjust) {
    for (auto img_in : imgs) {
        Mat img;
        if (img_in.channels() == 4)
            cvtColor(img_in, img, COLOR_BGRA2BGR);
        else if (img_in.channels() == 1)
            cvtColor(img_in, img, COLOR_GRAY2BGR);
        else img = img_in;
        std::vector<cv::Point> landmarks;
        Rect bbox(0, 0, img_in.cols, img_in.rows);
        align_impl(img, bbox, landmarks);
        if (!landmarks.size())
            return;

        Rect image_bbox = Rect(Point(0, 0), img.size());
        Rect rect = boundingRect(landmarks);
        rect &= image_bbox;
        if (adjust) {
            dlib::cv_image<dlib::bgr_pixel> dlib_img(img);
            dlib::array2d<dlib::bgr_pixel> out(face_size_[0], face_size_[1]);
            vector<dlib::point> avg_points, face_points;
            FaceDlibDetector::cv_point2dlib(avg_points_, avg_points);
            dlib::point_transform_affine trans;
            FaceDlibDetector::cv_point2dlib(landmarks, face_points);
            trans = dlib::find_similarity_transform(avg_points, face_points);
            dlib::transform_image(dlib_img, out, dlib::interpolate_bilinear(), trans);
            results.push_back(dlib::toMat(out).clone());
        } else {
            Mat face_image;
            resize(img(bbox), face_image, Size(face_size_[1], face_size_[0]));
            results.push_back(face_image);

        }
    }
}
void FaceAlignment::align_impl(const Mat &img, const Rect &bbox, std::vector<cv::Point> &landmarks) {
    dlib::cv_image<dlib::bgr_pixel> dlib_img(img);
    dlib::rectangle dlib_bbox;
    FaceDlibDetector::cv_rect2dlib(bbox, dlib_bbox);
    assert(dlib_bbox.right() <= img.cols);
    assert(dlib_bbox.bottom() <= img.rows);
    assert(dlib_bbox.left() > -1);
    assert(dlib_bbox.top() > -1);
    dlib::full_object_detection shape = sp_(dlib_img, dlib_bbox);
    landmarks.resize(0);
    if (shape.num_parts() < 1)
        return;
    landmarks.reserve(shape.num_parts());
    for (size_t i = 0; i < shape.num_parts(); ++i) {
        dlib::point &p = shape.part(i);
        landmarks.emplace_back(p.x(), p.y());
    }
}
FaceAlignment::FaceAlignment(const FaceAlignmentConfig &config) {
    dlib::deserialize(config.align_model) >> sp_;
    Mat avg_face = imread(config.align_deploy);
    Rect bbox(0, 0, avg_face.cols, avg_face.rows);
    Mat resized_img;
    std::vector<cv::Point> landmarks;
    face_size_ = config.face_size;

    if (face_size_.size() < 2) {
        face_size_.push_back(config.face_size[0]);
    }
    assert(face_size_.size() == 2);
    resize(avg_face, resized_img, Size(config.face_size[1], config.face_size[0]));
    align_impl(resized_img, bbox, landmarks);
    assert(landmarks.size());
    avg_points_ = landmarks;
}
}