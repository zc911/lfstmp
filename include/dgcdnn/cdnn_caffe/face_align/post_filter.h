#pragma once
#include "opencv2/opencv.hpp"
#include <vector>
using namespace std;
using namespace cv;

struct FaceScoreModel
{
	static float scoring(Mat& img, vector<Point2f> & shape);
	static float scoring(Mat_<float>& f);
};
