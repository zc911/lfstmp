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

<<<<<<< HEAD
    vector<Score> Rank(const Mat& image, const Rect& hotspot, const vector<CarRankFeature>& candidates)
    {
        CarRankFeature des;
        car_matcher_.ExtractDescriptor(image, des);
        LOG(INFO) << "image feature w(" << des.width_ << "), h(" << des.height_ << ")"; 

        float resize_rto = 600.0 / (float) max(image.cols, image.rows);
        int offset = (600 - resize_rto * image.cols) / 2;

        Rect hotspot_resized(hotspot);
        hotspot_resized.x *= resize_rto;
        hotspot_resized.y *= resize_rto;
        hotspot_resized.width *= resize_rto;
        hotspot_resized.height *= resize_rto;
        
//        hotspot_resized.x = 1.0 * (hotspot_resized.x - offset) / resize_rto;
//        hotspot_resized.y = 1.0 * (hotspot_resized.y - offset) / resize_rto;
//        hotspot_resized.width = 1.0 * hotspot_resized.width / resize_rto;
//        hotspot_resized.height = 1.0 * hotspot_resized.height / resize_rto;
        LOG(INFO) << "hotspot resized: " << hotspot_resized;

        t_profiler_matching_.Reset();
        vector<int> score = car_matcher_.ComputeMatchScore(des, hotspot_resized, candidates);
        t_profiler_str_ = "TotalMatching";
        t_profiler_matching_.Update(t_profiler_str_);
        
        vector<Score> topx(score.size());
        for (int i = 0; i < score.size(); i++)
        {
            topx[i] = Score(i, score[i]);
        }
        
        LOG(INFO)<< "Ranking finished, " <<t_profiler_matching_.getSmoothedTimeProfileString();
        return topx;
    }
=======
    vector<Score> rank(const Mat& image, const Rect& hotspot,
                       const vector<CarRankFeature>& candidates);
>>>>>>> origin
};

}

#endif //MATRIX_ENGINE_CAR_RANK_PROCESSOR_H_
