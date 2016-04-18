/*============================================================================
 * File Name   : feature_serializer.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#include <iterator>
#include <codec/base64.h>
#include "feature_serializer.h"

namespace dg {

FeatureSerializer::FeatureSerializer()
{

}

FeatureSerializer::~FeatureSerializer()
{

}


template<typename T>
void FeatureSerializer::ConvertToByte(T value, vector<uchar> &data)
{
    uchar *ptr = (uchar *) (&value);
    copy(ptr, ptr + sizeof(T), back_inserter(data));
}

template<typename T>
void FeatureSerializer::ConvertToValue(T *value, vector<uchar> data)
{
    uchar *ptr = (uchar *) value;
    for (int i = 0; i < sizeof(T); i++)
    {
        ptr[i] = data[i];
    }
}

string FeatureSerializer::FeatureSerialize(Mat des, Mat pos)
{
    float version = 1.0;

    vector<uchar> data;
    ConvertToByte(version, data);
    
    ConvertToByte((int) (des.dataend - des.datastart), data);
    ConvertToByte((int) (pos.dataend - pos.dataend), data);
    copy(des.datastart, des.dataend, back_inserter(data));
    copy(pos.datastart, pos.dataend, back_inserter(data));
    return Base64::Encode(data);
}

void FeatureSerializer::FeatureDeserialize(string str, Mat &des, Mat &pos)
{
    float version;
    int des_size, pos_size;

    vector<uchar> data;
    Base64::Decode(str, data);

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

    Mat des_p = Mat(des_size / 32, 32, 0, des_v.data());
    Mat pos_p = Mat(pos_size / (2 * sizeof(ushort)), 2, 2, pos_v.data());
    des_p.copyTo(des);
    pos_p.copyTo(pos);
}
}
