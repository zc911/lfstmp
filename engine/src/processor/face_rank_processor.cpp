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
//#include "codec/base64.h"

namespace dg {

FaceRankProcessor::FaceRankProcessor() {

}

FaceRankProcessor::~FaceRankProcessor() {
}

static float score_normalize(float euclid_dist) {
    const float alpha = -0.02;
    const float beta = 1.1;
    float r = alpha * euclid_dist + beta;

    if (r <= 0.0f) {
        r = 0.0f;
    } else if (r > 1.0f) {
        r = 1.0f;
    }

    return r;
}


bool FaceRankProcessor::process(Frame *frame) {

    FaceRankFrame *fframe = (FaceRankFrame *) frame;


    const RankCandidatesRepo &repo = RankCandidatesRepo::GetInstance();
    CDatabase &ranker = RankCandidatesRepo::GetInstance().GetFaceRanker();

    vector<float> &feature = fframe->datum_.feature_;

    int candidatesNum = fframe->max_candidates_;

    if(candidatesNum > ranker.GetTotalItems()){
        candidatesNum = ranker.GetTotalItems();
        LOG(WARNING) << "Candidate number exceeds the ranker database size " << candidatesNum << ":" << ranker.GetTotalItems() << endl;
        LOG(WARNING) << "We will use the top (size - 1) in default" << candidatesNum << endl;
    }
    if(candidatesNum <= 0){
        LOG(ERROR) << "Candidates id less than 0 " << endl;
        return false;
    }

    vector<CDatabase::DIST> results(candidatesNum);
    ranker.NearestN(feature.data(), candidatesNum, results.data());
    fframe->result_.clear();

//    cout << "input: " << endl;
//    for(auto v:feature){
//        cout << v << ",";
//    }
//    cout << endl;
    for (auto r: results) {
        Score score;
        score.index_ = r.id;
        score.score_ = score_normalize(r.dist);
//        score.score_ = r.dist;
//        vector<float> a(128);
//        ranker.RetrieveItemById(r.id, a.data());
//        cout <<  r.id << " " << r.dist << " : " << endl;
//        for(auto v:a){
//            cout << v << ",";
//        }
//        cout << endl;

//        score.score_ = score_normalize(r.dist);
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
