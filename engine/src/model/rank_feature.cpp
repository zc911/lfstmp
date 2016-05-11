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

    float version = 1.0;
    vector<uchar> data;
    ConvertToByte(version, data);

    const Mat &des = descriptor_;
    const Mat &pos = position_;
    ConvertToByte((int) (des.dataend - des.datastart), data);
    ConvertToByte((int) (pos.dataend - pos.dataend), data);
    copy(des.datastart, des.dataend, back_inserter(data));
    copy(pos.datastart, pos.dataend, back_inserter(data));
    return Base64::Encode(data);
}

bool CarRankFeature::Deserialize(string featureStr) {
    float version;
    int des_size, pos_size;

    vector<uchar> data;
    data.clear();
    Base64::Decode(featureStr, data);

    vector<uchar>::iterator it = data.begin();
    vector<uchar> version_v(it, it + sizeof(version));

    it += sizeof(version);
    vector<uchar> des_size_v(it, it + sizeof(des_size));

    it += sizeof(des_size);
    vector<uchar> pos_size_v(it, it + sizeof(des_size));

    ConvertToValue(&version, version_v);
    ConvertToValue(&des_size, des_size_v);
    ConvertToValue(&pos_size, pos_size_v);

    it += sizeof(pos_size);
    vector<uchar> des_v(it, it + des_size);

    it += des_size;
    vector<uchar> pos_v(it, it + pos_size);

    Mat des(des_size / 32, 32, 0, des_v.data());
    Mat pos(pos_size / (2 * sizeof(ushort)), 2, 2, pos_v.data());
    des.copyTo(descriptor_);
    pos.copyTo(position_);

    width_ = descriptor_.cols;
    height_ = descriptor_.rows;

    return true;
}

string FaceRankFeature::Serialize() const {
    return Base64::Encode(descriptor_);
}

bool FaceRankFeature::Deserialize(string featureStr) {
    descriptor_.clear();
    Base64::Decode(featureStr, descriptor_);
    return true;
}

