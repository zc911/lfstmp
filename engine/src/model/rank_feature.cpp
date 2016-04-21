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

string CarFeature::Serialize()
{
    float version = 1.0;
    vector<uchar> data;
    ConvertToByte(version, data);
    
    Mat &des = descriptor;
    Mat &pos = position;
    ConvertToByte((int) (des.dataend - des.datastart), data);
    ConvertToByte((int) (pos.dataend - pos.dataend), data);
    copy(des.datastart, des.dataend, back_inserter(data));
    copy(pos.datastart, pos.dataend, back_inserter(data));
    return Base64::Encode(data);
}

bool CarFeature::Deserialize(string featureStr)
{
    vector<uchar> data;
    Base64::Decode(featureStr, data);

    float version;
    int des_size, pos_size;
    int hpos = 0;
    vector<uchar> version_v(data.begin() + hpos,
            data.begin() + hpos + sizeof(version));
    hpos += sizeof(version);
    vector<uchar> des_size_v(data.begin() + hpos,
            data.begin() + hpos + sizeof(des_size));
    hpos += sizeof(des_size);
    vector<uchar> pos_size_v(data.begin() + hpos,
            data.begin() + hpos + sizeof(des_size));
    hpos += sizeof(pos_size);

    ConvertToValue(&version, version_v);
    ConvertToValue(&des_size, des_size_v);
    ConvertToValue(&pos_size, pos_size_v);

    vector<uchar> des_v(data.begin() + hpos, data.begin() + hpos + des_size);
    hpos += des_size;

    vector<uchar> pos_v(data.begin() + hpos, data.begin() + hpos + pos_size);
    hpos += pos_size;

    Mat des_p = Mat(des_size / (32 * sizeof(ushort)), 32, 0, des_v.data());
    Mat pos_p = Mat(pos_size / (2 * sizeof(ushort)), 2, 2, pos_v.data());
    des_p.copyTo(descriptor);
    pos_p.copyTo(position);


    // float version;
    // int des_size, pos_size;

    // vector<uchar> data;
    // data.clear();
    // Base64::Decode(featureStr, data);

    // vector<uchar>::iterator it = data.begin();    
    // vector<uchar> version_v(it, it + sizeof(version));

    // it += sizeof(version);
    // vector<uchar> des_size_v(it, it + sizeof(des_size));

    // it += sizeof(des_size);
    // vector<uchar> pos_size_v(it, it + sizeof(des_size));

    // ConvertToValue(&version, version_v);
    // ConvertToValue(&des_size, des_size_v);
    // ConvertToValue(&pos_size, pos_size_v);

    // it += sizeof(pos_size);
    // vector<uchar> des_v(it, it + des_size);

    // it += des_size;
    // vector<uchar> pos_v(it, it + pos_size);

    // Mat des(des_size / 32, 32, 0, des_v.data());
    // Mat pos(pos_size / (2 * sizeof(ushort)), 2, 2, pos_v.data());
    // des.copyTo(descriptor);
    // pos.copyTo(position);

    width = descriptor.cols;
    height = descriptor.rows;

            int des_s = 0, pos_s = 0, str_s = 0;
            for(uchar u : data)
            {
                str_s += u;
            }

            for(int i = 0; i < descriptor.rows; i ++)
            {
                for(int j = 0; j < descriptor.cols; j ++)
                {
                    des_s += descriptor.at<ushort>(i, j);
                }
            }
            for(int i = 0; i < position.rows; i ++)
            {
                for(int j = 0; j < position.cols; j ++)
                {
                    pos_s += position.at<ushort>(i, j);
                }
            }

    LOG(INFO) << "feature: w(" << width << "), h(" << height << "), length: " << descriptor.size() << ", size: " << str_s;
    LOG(INFO) << descriptor.rows << ":" << descriptor.cols << ":" << des_s;
    LOG(INFO) << position.rows << ":" << position.cols << ":" << pos_s;

    return true;
}

string FaceFeature::Serialize()
{
    return Base64::Encode(descriptor);
}

bool FaceFeature::Deserialize(string featureStr)
{
    descriptor.clear();
    Base64::Decode(featureStr, descriptor);
    return true;
}

