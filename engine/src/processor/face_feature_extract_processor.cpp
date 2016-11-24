/*============================================================================
 * File Name   : face_feature_extract_processor.cpp
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年4月21日 下午3:44:11
 * Description :
 * ==========================================================================*/
#include "processor/face_feature_extract_processor.h"
#include "processor_helper.h"
#include "util/convert_util.h"

using namespace dgvehicle;
namespace dg {

FaceFeatureExtractProcessor::FaceFeatureExtractProcessor() {
    extractor_ = AlgorithmFactory::GetInstance()->CreateFaceFeatureExtractor();
}

FaceFeatureExtractProcessor::~FaceFeatureExtractProcessor() {
    if (extractor_)
        delete extractor_;
    to_processed_.clear();
}

bool FaceFeatureExtractProcessor::process(Frame *frame) {

    int size = frame->objects().size();

    for (int i = 0; i < size; ++i) {
        Object *obj = (frame->objects())[i];
        if (obj && obj->type() == OBJECT_FACE) {
            Face *face = static_cast<Face *>(obj);
            Rect rect;
            rect = face->detection().box;

            Mat img = frame->payload()->data();
            Mat cut = img(rect);

            vector<Mat> imgs;
            imgs.push_back(cut);
            vector<dgvehicle::FaceRankFeature> features = extractor_->Extract(imgs);
            FaceRankFeature feature = ConvertDgvehicleFaceRankFeature(features[0]);
            face->set_feature(feature);
        } else {
            DLOG(WARNING) << "Object is not type of face: " << obj->id() << endl;
        }

    }
    return true;
}

bool FaceFeatureExtractProcessor::process(FrameBatch *frameBatch) {


    for (int i = 0; i < to_processed_.size(); ++i) {
        Object *obj = to_processed_[i];
        Face *face = static_cast<Face *>(obj);
        performance_++;

        vector<Mat> imgs;
        imgs.push_back(face->image());
        vector<dgvehicle::FaceRankFeature> features = extractor_->Extract(imgs);
        FaceRankFeature feature = ConvertDgvehicleFaceRankFeature(features[0]);
        face->set_feature(feature);
    }

    return true;
}


bool FaceFeatureExtractProcessor::RecordFeaturePerformance() {

    return RecordPerformance(FEATURE_FACE_EXTRACT, performance_);

}
bool FaceFeatureExtractProcessor::beforeUpdate(FrameBatch *frameBatch) {
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
        } else {
            itr++;

        }
    }
    return true;
}
} /* namespace dg */
