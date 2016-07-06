#include "car_feature_extractor.h"
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;

namespace dg {

CarFeatureExtractor::CarFeatureExtractor() {
    orb_ = cv::ORB(CAR_FEATURE_ORB_ROWS_MAX);
    max_resize_size_ = 300;
}

void CarFeatureExtractor::ExtractDescriptor(const cv::Mat &img,
                                            CarRankFeature &des) {

    des.height_ = img.rows;
    des.width_ = img.cols;
    cv::Mat resize_img;
    cv::Size new_size;
    calcNewSize(des.height_, des.width_, new_size);

    if (img.channels() != 3)
        LOG(WARNING) << "Color image is required.";
    if ((img.rows < 10) || (img.cols < 10))
        LOG(WARNING) << "Image needs to be larger than 10*10 to extract enough feature.";

    resize(img, resize_img, new_size);

    vector<cv::KeyPoint> key_point;
    cv::Mat descriptor;

    orb_(resize_img, cv::Mat(), key_point, descriptor);

    if (key_point.size() < 50)
        LOG(WARNING) << "Not enough feature extracted.";

    descriptor.copyTo(des.descriptor_);

    des.position_ = cv::Mat::zeros(key_point.size(), 2, CV_16UC1);
    for (int i = 0; i < key_point.size(); i++) {
        des.position_.at<ushort>(i, 0) = ((ushort) key_point[i].pt.x);
        des.position_.at<ushort>(i, 1) = ((ushort) key_point[i].pt.y);
    }

}

void CarFeatureExtractor::calcNewSize(const ushort &ori_height,
                                      const ushort &ori_width,
                                      cv::Size &new_size) {
    float resize_rto = max(ori_height, ori_width);
    resize_rto = ((float) max_resize_size_) / resize_rto;
    new_size.height = resize_rto * ori_height;
    new_size.width = resize_rto * ori_width;

}
}
