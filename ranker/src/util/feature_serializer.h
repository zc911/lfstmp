/*============================================================================
 * File Name   : feature_serializer.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/
#ifndef MATRIX_RANKER_FEATURE_SERIALIZER_H_
#define MATRIX_RANKER_FEATURE_SERIALIZER_H_

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

namespace dg {
class FeatureSerializer
{
public:
    FeatureSerializer();
    virtual ~FeatureSerializer();

    string FeatureSerialize(Mat des, Mat pos);
    void FeatureDeserialize(string feature, Mat &des, Mat &pos);

private:
    template<typename T>
    void ConvertToByte(T value, vector<uchar> &data);

    template<typename T>
    void ConvertToValue(T *value, vector<uchar> data);
};
}

 #endif //MATRIX_RANKER_FEATURE_SERIALIZER_H_