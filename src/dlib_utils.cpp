#include "dlib_utils.h"

using namespace std;
using namespace cv;
namespace DGFace{

void dlib_point2cv(const vector<dlib::point> &input, vector<cv::Point> &output) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const dlib::point &point = input[i];
        output[i] = Point(point.x(), point.y());
    }
}

void dlib_rect2cv(const dlib::rectangle &input, Rect &output) {
    output = Rect(input.left(), input.top(), input.width(), input.height());
}

void cv_point2dlib(const vector<cv::Point> &input, vector<dlib::point> &output) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const cv::Point &point = input[i];
        output[i] = dlib::point(point.x, point.y);
    }
}

void cv_rect2dlib(const Rect &input, dlib::rectangle &output) {
    output = dlib::rectangle(input.x, input.y, input.x + input.width, input.y + input.height);
}

}