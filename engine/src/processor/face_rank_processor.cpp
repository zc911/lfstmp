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
namespace dg {

FaceRankProcessor::FaceRankProcessor() {
    ranker_ = new FaceRanker();
}

FaceRankProcessor::~FaceRankProcessor() {
    if (ranker_)
        delete ranker_;
}

bool FaceRankProcessor::process(Frame *frame) {
    FaceRankFrame *fframe = (FaceRankFrame *) frame;
    fframe->result_ = ranker_->Rank(fframe->datum_, fframe->hotspots_,
                                    fframe->candidates_);
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
