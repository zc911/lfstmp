#pragma once
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
/* This is the implementation of the classic POSIT algorithm for camera pose estimation*/
const float epsilon =0.00001f;
//const double PI =3.14159265358979323846;
const double pi_over_2 = 1.57079632679489661923;
const double pi_over_4 = 0.78539816339744830962;
const double RAD_TO_DEG_FACTOR =57.2957795;
const double DEG_TO_RAD_FACTOR =0.01745329;

//implementation of the classic POSIT algorithm 
void modernPosit(Mat_<float>& imagePoints, Mat_<float>& objectPoints, float focalLength, Point2f center,Mat_<float>& rot, Mat_<float>& trans);
void RotationMatrixFromEulerAngle(Mat_<float> &matrix, const float roll,const float pitch,const float yaw);
void RotationMatrixToEulerAngle(const Mat &rot_matrix, float& roll, float& pitch, float& yaw);
