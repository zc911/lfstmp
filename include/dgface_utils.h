#ifndef _DGFACESDK_UTILS_H_
#define _DGFACESDK_UTILS_H_
#include "watch_dog.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include "face_inf.h"

namespace DGFace {
// decrypt config file
int getConfigContent(std::string file, bool is_encrypt, std::string& content);
void addNameToPath(const std::string& model_dir, const std::string& name, std::string& appended_dir);

//
// draw function
void drawRotatedRect(cv::Mat& draw_img, const cv::RotatedRect& rot_bbox);
void drawDetectionResult(cv::Mat& draw_img, const DetectResult& det_result, bool display_score);
void drawLandmarks(cv::Mat& draw_img, const AlignResult &align_result);

//
// save and load function
bool saveFeature(const std::string& fea_log, const std::vector<RecogResult>& recog_results);
bool loadFeature(const std::string& fea_log, std::vector<RecogResult>& recog_results);

//
// convert function
void cvtPoint2iToPoint2f(const std::vector<cv::Point>& pts_2i, std::vector<cv::Point2f>& pts_2f);
void cvtPoint2fToPoint2i(const std::vector<cv::Point2f>& pts_2f, std::vector<cv::Point>& pts_2i);

void cvtLandmarks(const std::vector<cv::Point2f>& src_landmarks, std::vector<Point_2d_f>& dst_landmarks);
void cvtLandmarks(const std::vector<Point_2d_f>& src_landmarks, std::vector<cv::Point2f>& dst_landmarks);
void cvtLandmarks(const std::vector<cv::Point2f>& src_landmarks, std::vector<double>& dst_landmarks);
void cvtLandmarks(const std::vector<double>& src_landmarks, std::vector<cv::Point2f>& dst_landmarks);
void cvtLandmarks(const std::vector<Point_2d_f>& src_landmarks, std::vector<double>& dst_landmarks);
void cvtLandmarks(const std::vector<double>& src_landmarks, std::vector<Point_2d_f>& dst_landmarks);

//
// evaluation(AP, mAP)

float computeAP(const std::vector<float>& scores, const std::vector<bool>& trues);
float computeMAP(const std::vector<std::vector<float> >& score_vec, const std::vector<std::vector<bool> >& true_vec);
float computeMAP(const std::vector<std::vector<float> >& score_vec, const std::vector<std::vector<bool> >& true_vec, std::vector<float>& AP_vec);

}
#endif
