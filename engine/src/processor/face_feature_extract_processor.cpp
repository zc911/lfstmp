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
namespace dg {

FaceFeatureExtractProcessor::FaceFeatureExtractProcessor(
    const FaceFeatureExtractor::FaceFeatureExtractorConfig &config) {
    extractor_ = new FaceFeatureExtractor(config);
}

FaceFeatureExtractProcessor::~FaceFeatureExtractProcessor() {
    if (extractor_)
        delete extractor_;
    to_processed_.clear();
}

bool FaceFeatureExtractProcessor::process(Frame *frame) {

    int size = frame->objects().size();

    vector<Mat> imgs;
    for (int i = 0; i < size; ++i) {
        Object *obj = (frame->objects())[i];
        if (obj && obj->type() == OBJECT_FACE) {
            Face *face = static_cast<Face *>(obj);
            Rect rect;
            rect = face->detection().box;

            Mat img = frame->payload()->data();
            Mat cut = img(rect);

            imgs.push_back(cut);

        } else {
            DLOG(WARNING) << "Object is not type of face: " << obj->id() << endl;
        }

    }
    vector<FaceRankFeature> features = extractor_->Extract(imgs);
    if (size != features.size()) {
        LOG(ERROR) << "Face image size not equals to feature size: " << size << ":" << features.size() << endl;
        return false;
    }

    for (int i = 0; i < size; ++i) {
        Object *obj = (frame->objects())[i];
        if (obj && obj->type() == OBJECT_FACE) {
            Face *face = static_cast<Face *>(obj);

            FaceRankFeature feature = features[i];
            face->set_feature(feature);
        }
    }

    return true;
}

bool FaceFeatureExtractProcessor::process(FrameBatch *frameBatch) {

    vector<Mat> imgs;
    for (auto obj : to_processed_) {
        Face *face = static_cast<Face *>(obj);
        imgs.push_back(face->image());
        performance_++;
    }

    if (imgs.size() == 0) {
        return false;
    }

    vector<FaceRankFeature> features = extractor_->Extract(imgs);
    if (features.size() != imgs.size()) {
        LOG(ERROR) << "Face image size not equals to feature size: " << imgs.size() << ":" << features.size() << endl;
        return false;
    }

    for (int i = 0; i < features.size(); ++i) {
        FaceRankFeature feature = features[i];
        Face *face = (Face *) to_processed_[i];
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
