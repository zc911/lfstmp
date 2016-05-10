/*============================================================================
 * File Name   : face_feature_extract_processor.cpp
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年4月21日 下午3:44:11
 * Description : 
 * ==========================================================================*/
#include "processor/face_feature_extract_processor.h"

namespace dg {

FaceFeatureExtractProcessor::FaceFeatureExtractProcessor(
        const FaceFeatureExtractor::FaceFeatureExtractorConfig &config) {
    extractor_ = new FaceFeatureExtractor(config);
}

FaceFeatureExtractProcessor::~FaceFeatureExtractProcessor() {
    delete extractor_;
}

void FaceFeatureExtractProcessor::Update(Frame *frame) {
    int size = frame->objects().size();

    for (int i = 0; i < size; ++i) {
        Object * obj = frame->get_object(i);
        if (obj->type() == OBJECT_FACE) {
            Face *face = static_cast<Face*>(obj);
            Rect rect;
            rect = face->detection().box;

            Mat img = frame->payload()->data();
            Mat cut = img(rect);

            vector<Mat> imgs;
            imgs.push_back(cut);
            vector<FaceRankFeature> features = extractor_->Extract(imgs);
            FaceRankFeature feature = features[0];
            face->set_feature(feature);
        } else {
            DLOG(WARNING)<< "Object is not type of face: " << obj->id() << endl;
        }

    }

    Proceed(frame);
}

void FaceFeatureExtractProcessor::Update(FrameBatch *frameBatch) {
    for (int i = 0; i < frameBatch->frames().size(); ++i) {
        Frame *frame = frameBatch->frames()[i];
        int size = frame->objects().size();

        for (int i = 0; i < size; ++i) {
            Object * obj = frame->get_object(i);
            if (obj->type() == OBJECT_FACE) {

                Face *face = static_cast<Face*>(obj);
                Rect rect = face->detection().box;
                Mat img = frame->payload()->data();
                Mat cut = img(rect);

                vector<Mat> imgs;
                imgs.push_back(cut);
                vector<FaceRankFeature> features = extractor_->Extract(imgs);
                FaceRankFeature feature = features[0];
                face->set_feature(feature);
            } else {
                DLOG(WARNING)<< "Object is not type of face: " << obj->id() << endl;
            }

        }
    }
}

} /* namespace dg */
