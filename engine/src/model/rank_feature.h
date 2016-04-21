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
#include <glog/logging.h>
#include <opencv2/core/core.hpp>

using namespace std;

namespace dg 
{
class Score
{
public:
    int index;
    float score;

    //sortable: score[desc] >> index[asc]
    bool operator<(const Score& right) const
    {
        return score != right.score ? (score > right.score) : (index < right.index);
    }
};

class RankFeature 
{
public:
    virtual string Serialize() {};
    virtual bool Deserialize(string featureStr) {};

protected:
    template<typename T>
    static void ConvertToByte(T value, vector<uchar> &data)
    {
        uchar *ptr = (uchar *) (&value);
        copy(ptr, ptr + sizeof(T), back_inserter(data));
    }

    template<typename T>
    static void ConvertToValue(T *value, vector<uchar> data)
    {
        uchar *ptr = (uchar *) value;
        for (int i = 0; i < sizeof(T); i++)
        {
            ptr[i] = data[i];
        }
    }
};

class CarFeature final : public RankFeature
{
public:
    int width;
    int height;
    cv::Mat descriptor;
    cv::Mat position;

    virtual string Serialize() override;
    virtual bool Deserialize(string featureStr) override;
};

class FaceFeature final : public RankFeature
{
public:
    std::vector<float> descriptor;

    virtual string Serialize() override;
    virtual bool Deserialize(string featureStr) override;
};


}

 #endif //MATRIX_RANKER_FEATURE_SERIALIZER_H_