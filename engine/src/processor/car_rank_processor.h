/*============================================================================
 * File Name   : car_rank_processor.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_ENGINE_CAR_RANK_PROCESSOR_H_
#define MATRIX_ENGINE_CAR_RANK_PROCESSOR_H_

#include <glog/logging.h>

#include "processor.h"
#include "model/frame.h"
#include "model/rank_feature.h"
#include "alg/car_matcher.h"
#include "alg/car_feature_extractor.h"

#include "timing_profiler.h"

namespace dg {

class CarRankProcessor : public Processor {
 public:
    CarRankProcessor();
    virtual ~CarRankProcessor();
    virtual void Update(Frame *frame);
    virtual void Update(FrameBatch *frameBatch);

    virtual bool checkOperation(Frame *frame);

    virtual bool checkStatus(Frame *frame);

 private:
    string t_profiler_str_;
    TimingProfiler t_profiler_matching_;
    CarMatcher car_matcher_;
    CarFeatureExtractor car_feature_extractor_;

    vector<Score> rank(const Mat& image, const Rect& hotspot,
                       const vector<CarRankFeature>& candidates);
};

}

#endif //MATRIX_ENGINE_CAR_RANK_PROCESSOR_H_
