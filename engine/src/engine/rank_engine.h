/*============================================================================
 * File Name   : rank_engine.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_ENGINE_RANK_ENGINE_H_
#define MATRIX_ENGINE_RANK_ENGINE_H_

#include <glog/logging.h>

#include "engine.h"
#include "timing_profiler.h"
#include "model/frame.h"
#include "model/model.h"
#include "model/ringbuffer.h"
#include "model/rank_feature.h"
#include "processor/car_rank_processor.h"
#include "processor/face_rank_processor.h"


namespace dg {

class RankEngine : public Engine {
public:
    RankEngine() : Engine() {}
    virtual ~RankEngine() {}

    virtual void Process() override {}

    virtual int Stop() override { return 0; }

    virtual int Release() override { return 0; }
};

class CarRankEngine : public RankEngine {
public:
    CarRankEngine() : id_(0) {}
    virtual ~CarRankEngine() {}

    vector<Score> Rank(const Mat& image, const Rect& hotspot, const vector<CarFeature>& candidates)
    {
        vector<Rect> hotspots;
        hotspots.push_back(hotspot);
        CarRankFrame f(id_++, image, hotspots, candidates);
        processor_.Update(&f);
        return f.result_;
    }

private:
    Identification id_;
    CarRankProcessor processor_;
};


class FaceRankEngine : public RankEngine {
public:
    FaceRankEngine() : id_(0) {}
    virtual ~FaceRankEngine() {}

    vector<Score> Rank(const Mat& image, const Rect& hotspot, const vector<FaceFeature>& candidates)
    {
        vector<Rect> hotspots;
        hotspots.push_back(hotspot);
        FaceRankFrame f(id_++, image, hotspots, candidates);
        processor_.Update(&f);
        return f.result_;
    }

private:
    Identification id_;
    FaceRankProcessor processor_;
};

}

#endif //MATRIX_ENGINE_RANK_ENGINE_H_