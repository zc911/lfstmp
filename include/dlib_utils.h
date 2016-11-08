#ifndef _DGFACESDK_DLIB_UTILS_H_
#define _DGFACESDK_DLIB_UTILS_H_

#include "common.h"
#include "../dlib/geometry.h"
namespace DGFace{

void dlib_point2cv(const std::vector<dlib::point> &input, std::vector<cv::Point> &output);

void dlib_rect2cv(const dlib::rectangle &input, cv::Rect &output);

void cv_point2dlib(const std::vector<cv::Point> &input, std::vector<dlib::point> &output);

void cv_rect2dlib(const cv::Rect &input, dlib::rectangle &output);
}
#endif

