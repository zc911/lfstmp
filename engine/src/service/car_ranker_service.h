/*============================================================================
 * File Name   : car_ranker_service.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_RANKER_CAR_RANKER_SERVICE_H_
#define MATRIX_RANKER_CAR_RANKER_SERVICE_H_

#include <glog/logging.h>
#include "alg/car_matcher.h"
#include "ranker_service.h"
#include "timing_profiler.h"

namespace dg 
{

class CarRanker final : public Ranker<CarFeature>
{
public:
    CarRanker();
    virtual ~CarRanker();

private:
    string t_profiler_str_;
    TimingProfiler t_profiler_matching_;
    CarMatcher car_matcher_;

    virtual vector<Score> Rank(const Mat& image, const Rect& hotspot, const vector<CarFeature>& candidates) override;
};

}

#endif //MATRIX_RANKER_CAR_RANKER_SERVICE_H_