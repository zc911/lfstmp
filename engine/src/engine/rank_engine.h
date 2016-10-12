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
#include "processor/processor.h"
#include "model/model.h"
#include "model/rank_feature.h"
#include "config.h"
#include "engine_config_value.h"

#define RANKER_MAXIMUM 10000

namespace dg {

class RankEngine {
public:
    RankEngine(const Config &config) {
        SetMaxCandidatesSize(config);
    }
    virtual ~RankEngine() { };
    void SetMaxCandidatesSize(const Config &config) {
        max_candidates_size_ = min(RANKER_MAXIMUM, (int) config.Value(ADVANCED_RANKER_MAXIMUM));
    }
    int GetMaxCandidatesSize() {
        return max_candidates_size_;
    }
    int max_candidates_size_;

};


class SimpleRankEngine: public RankEngine {
public:
    SimpleRankEngine(const Config &config);
    virtual ~SimpleRankEngine();

    void RankCar(CarRankFrame *frame);
    void RankFace(FaceRankFrame *frame);
    void AddFeatures(FeaturesFrame *frame);

private:
    Identification id_;
    Processor *car_ranker_ = NULL;
    bool enable_ranker_face_;
    bool enable_ranker_car_;
    Processor *face_ranker_ = NULL;
};

// class FaceRankEngine: public RankEngine {

// public:
//     FaceRankEngine(const Config &config);
//     virtual ~FaceRankEngine();
//     vector<Score> Rank(const Mat &image, const Rect &hotspot,
//                        const vector<FaceRankFeature> &candidates);
// private:
//     void init(const Config &config);
//     Identification id_;
//     Processor *detector_;
//     Processor *extractor_;
//     Processor *ranker_;
// };

}

#endif //MATRIX_ENGINE_RANK_ENGINE_H_
