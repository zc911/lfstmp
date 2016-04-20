/*============================================================================
 * File Name   : car_ranker_service.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#include <glog/logging.h>

#include "car_ranker_service.h"

using namespace dg;

CarRanker::CarRanker()
{
}

CarRanker::~CarRanker()
{
}

vector<Score> CarRanker::Rank(const Mat& image, const Rect& hotspot, const vector<CarFeature>& candidates)
{
    CarFeature des;
    car_matcher_.extract_descriptor(image, des);

    float resize_rto = 600.0 / (float) max(image.cols, image.rows);
    int offset = (600 - resize_rto * image.cols) / 2;

    Rect hotspot_resized(hotspot);
    hotspot_resized.x = 1.0 * (hotspot_resized.x - offset) / resize_rto;
    hotspot_resized.y = 1.0 * (hotspot_resized.y - offset) / resize_rto;
    hotspot_resized.width = 1.0 * hotspot_resized.width / resize_rto;
    hotspot_resized.height = 1.0 * hotspot_resized.height / resize_rto;

    t_profiler_matching_.Reset();
    vector<int> score = car_matcher_.compute_match_score(des, hotspot_resized, candidates);
    t_profiler_str_ = "TotalMatching";
    t_profiler_matching_.Update(t_profiler_str_);
    
    vector<Score> topx(score.size());
    for (int i = 0; i < score.size(); i++)
    {
        Score cs;
        cs.index = i;
        cs.score = score[i];
        topx[i] = cs;
    }
    
    LOG(INFO)<< "Ranking finished, " <<t_profiler_matching_.getSmoothedTimeProfileString();
    return topx;
}
