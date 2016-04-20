/*============================================================================
 * File Name   : ranker_service.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_RANKER_RANKER_SERVICE_H_
#define MATRIX_RANKER_RANKER_SERVICE_H_

#include <vector>
#include <opencv2/core/core.hpp>

#include "alg/feature.h"

using namespace cv;
using namespace std;

namespace dg 
{

template <typename F>
class Ranker
{
static_assert(std::is_base_of<Feature, F>::value, "F must derive from Feature");

public:
    F Deserialize(string featureStr) const
    {
        F feature;
        feature.Deserialize(featureStr);
        return feature;
    }

    string Serialize(F& feature) const
    {
        return feature.Serialize();
    }

    virtual vector<Score> Rank(const Mat& image, const Rect& hotspot, const vector<F>& candidates);
};

}

#endif //MATRIX_RANKER_RANKER_SERVICE_H_