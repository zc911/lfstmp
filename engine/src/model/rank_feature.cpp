/*============================================================================
 * File Name   : feature_serializer.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#include <iterator>
#include <opencv2/core/core.hpp>
#include "rank_feature.h"
#include "codec/base64.h"

using namespace cv;
using namespace dg;

string CarRankFeature::Serialize() const {

    if (descriptor_.cols == 0 || position_.cols == 0 || descriptor_.rows == 0
        || position_.rows == 0) {
        return "";
    }

    float version = 2.0;
    vector<uchar> data;
    ConvertToByte(version, data);
    ConvertToByte(width_, data);
    ConvertToByte(height_, data);

    const Mat &des = descriptor_;
    const Mat &pos = position_;
    ConvertToByte((int) (des.dataend - des.datastart), data);
    ConvertToByte((int) (pos.dataend - pos.datastart), data);
    copy(des.datastart, des.dataend, back_inserter(data));
    copy(pos.datastart, pos.dataend, back_inserter(data));
    return Base64::Encode(data);
}

bool CarRankFeature::Deserialize(string featureStr) {
    float version;
    int des_size, pos_size;
    ushort width, height;

    vector<uchar> data;
    data.clear();
    Base64::Decode(featureStr, data);

    vector<uchar>::iterator it = data.begin();

    vector<uchar> version_v(it, it + sizeof(version));
    it += sizeof(version);

    vector<uchar> width_v(it, it + sizeof(width));
    it += sizeof(width);

    vector<uchar> height_v(it, it + sizeof(height));
    it += sizeof(height);

    vector<uchar> des_size_v(it, it + sizeof(des_size));
    it += sizeof(des_size);

    vector<uchar> pos_size_v(it, it + sizeof(des_size));

    ConvertToValue(&version, version_v);
    ConvertToValue(&width, width_v);
    ConvertToValue(&height, height_v);
    ConvertToValue(&des_size, des_size_v);
    ConvertToValue(&pos_size, pos_size_v);

    DLOG(INFO) << "Feature version: " << version << endl;
    DLOG(INFO) << "Des size: " << des_size << ", and Pos size: " << pos_size << endl;

    // this shit check validates the input data
    if (des_size <= 0 || pos_size <= 0 || des_size % CAR_FEATURE_ORB_COLS_MAX != 0 || pos_size % 2 != 0
        || des_size % CAR_FEATURE_ORB_COLS_MAX > CAR_FEATURE_ORB_ROWS_MAX
        || it + des_size >= data.end() || it + pos_size >= data.end() || it + des_size + pos_size >= data.end()) {

        LOG(ERROR) << "Feature version:" << version << endl;
        LOG(ERROR) << "Des size: " << des_size << ", and Pos size: " << pos_size << endl;
        LOG(ERROR) << "Feature string invalid" << endl;
        return false;
    }

    it += sizeof(pos_size);
    vector<uchar> des_v(it, it + des_size);
    it += des_size;

    vector<uchar> pos_v(it, it + pos_size);

    Mat des(des_size / 32, 32, CV_8UC1, des_v.data());
    Mat pos(pos_size / 2, 2, CV_16UC1, pos_v.data());
    des.copyTo(descriptor_);
    pos.copyTo(position_);

    width_ = width;
    height_ = height;

    return true;
}

string FaceRankFeature::Serialize() const {
    if (descriptor_.size() == 0) {
        return "";
    }
    return Base64::Encode(descriptor_);
}

bool FaceRankFeature::Deserialize(string featureStr) {
    descriptor_.clear();
    Base64::Decode(featureStr, descriptor_);
    return true;
}

