/*============================================================================
 * File Name   : face_rank_processor.cpp
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年4月28日 下午4:09:47
 * Description :
 * ==========================================================================*/
#include "processor/face_rank_processor.h"
#include "processor_helper.h"
#include "io/rank_candidates_repo.h"
#include "alg/rank/database.h"

namespace dg {

FaceRankProcessor::FaceRankProcessor() {

}

FaceRankProcessor::~FaceRankProcessor() {
}

bool FaceRankProcessor::process(Frame *frame) {

    FaceRankFrame *fframe = (FaceRankFrame *) frame;


    const RankCandidatesRepo &repo = RankCandidatesRepo::GetInstance();
    CDatabase &ranker = RankCandidatesRepo::GetInstance().GetFaceRanker();

    vector<float> &feature = fframe->datum_.feature_;

    int candidatesNum = 10;
    vector<CDatabase::DIST> results(candidatesNum);


    ranker.NearestN(feature.data(), candidatesNum, results.data());
    fframe->result_.clear();
    for (auto r: results) {
        Score score;
        score.index_ = r.id;
        score.score_ = r.dist;
        fframe->result_.push_back(score);
    }

    performance_++;

    return true;

}
bool FaceRankProcessor::RecordFeaturePerformance() {

    return RecordPerformance(FEATURE_FACE_RANK, performance_);

}
bool FaceRankProcessor::beforeUpdate(FrameBatch *frameBatch) {
#if RELEASE
    if (performance_ > RECORD_UNIT) {
        if (!RecordFeaturePerformance()) {
            return false;
        }
    }
#endif

    return true;
}
} /* namespace dg */
