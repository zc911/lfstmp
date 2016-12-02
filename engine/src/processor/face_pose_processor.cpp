/*============================================================================
 * File Name   : face_feature_extract_processor.cpp
 * Author      : jiajiachen@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年4月21日 下午3:44:11
 * Description :
 * ==========================================================================*/
//#include <alg/feature/face_alignment.h>
#include "processor/face_pose_processor.h"
#include "processor_helper.h"
namespace dg {

FacePoseProcessor::FacePoseProcessor(const FacePoseConfig &config) {
    fp_ = new DGFace::PoseQuality();
}

FacePoseProcessor::~FacePoseProcessor() {
    if (fp_)
        delete fp_;
}
static void draw_landmarks(Mat& img, const DGFace::AlignResult &align_result) {
    auto &landmarks = align_result.landmarks;
    for (auto pt = landmarks.begin(); pt != landmarks.end(); ++pt)
    {
        circle(img, *pt, 2, Scalar(0, 255, 0), -1);
    }
}
bool FacePoseProcessor::process(FrameBatch *frameBatch) {
    if (!fp_)
        return false;
    for (vector<Object *>::iterator itr = to_processed_.begin(); itr != to_processed_.end(); itr++) {

        Mat img = ((Face *)(*itr))->image();
        DGFace::AlignResult ar = ((Face *)(*itr))->get_align_result();

        vector<float> scores = fp_->quality(ar);
        ((Face *)(*itr))->set_pose(scores);
        performance_++;
    }

    return true;
}

bool FacePoseProcessor::RecordFeaturePerformance() {

    return RecordPerformance(FEATURE_FACE_EXTRACT, performance_);

}
bool FacePoseProcessor::beforeUpdate(FrameBatch *frameBatch) {
#if DEBUG
#else    //#if RELEASE
    if (performance_ > RECORD_UNIT) {
        if (!RecordFeaturePerformance()) {
            return false;
        }
    }
#endif
    to_processed_.clear();
    to_processed_ = frameBatch->CollectObjects(OPERATION_FACE_FEATURE_VECTOR);
    for (vector<Object *>::iterator itr = to_processed_.begin();
         itr != to_processed_.end();) {
        if ((*itr)->type() != OBJECT_FACE) {
            itr = to_processed_.erase(itr);
        } else if (((Face *)(*itr))->image().rows == 0 || ((Face *)(*itr))->image().cols == 0) {
            itr = to_processed_.erase(itr);
        } else {
            itr++;
        }
    }
    //LOG(INFO) << to_processed_.size();
    return true;
}
} /* namespace dg */