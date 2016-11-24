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
#include "util/convert_util.h"

using namespace dgvehicle;
namespace dg {

FaceRankProcessor::FaceRankProcessor() {
    ranker_ = AlgorithmFactory::GetInstance()->CreateFaceRanker();
}

FaceRankProcessor::~FaceRankProcessor() {
    if (ranker_)
        delete ranker_;
}

bool FaceRankProcessor::process(Frame *frame) {
    FaceRankFrame *fframe = (FaceRankFrame *) frame;

    vector<dgvehicle::FaceRankFeature> faceCandidates;
    dgvehicle::FaceRankFeature faceDatum = ConvertToDgvehicleFaceRankFeature(fframe->datum_);
    for (auto candidate : fframe->candidates_) {
        faceCandidates.push_back(ConvertToDgvehicleFaceRankFeature(candidate));
    }
    vector<dgvehicle::Score> results = ranker_->Rank(faceDatum, fframe->hotspots_, faceCandidates);
    fframe->result_.clear();
    for (auto result : results) {
        fframe->result_.push_back(ConvertDgvehicleScore(result));
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
