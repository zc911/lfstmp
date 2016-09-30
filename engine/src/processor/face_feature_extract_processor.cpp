/*============================================================================
 * File Name   : face_feature_extract_processor.cpp
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年4月21日 下午3:44:11
 * Description :
 * ==========================================================================*/
//#include <alg/feature/face_alignment.h>
#include <alg/feature/face_feature_extractor.h>
#include "processor/face_feature_extract_processor.h"
#include "processor_helper.h"
namespace dg {

FaceFeatureExtractProcessor::FaceFeatureExtractProcessor(
    const FaceFeatureExtractorConfig &config, const FaceAlignmentConfig &faConfig) {
    switch (config.method) {
    case CNNRecog:
        recognition_ = new DGFace::CNNRecog(config.deploy_file, config.model_file, config.layer_name, config.mean, config.pixel_scale, config.use_GPU);
        break;
    case LBPRecog: {
        int radius = 1;
        int neighbors = 8;
        int grid_x = 8;
        int grid_y = 8;
        recognition_ = new DGFace::LbpRecog(radius, neighbors, grid_x, grid_y);
        break;
    }
    }
    switch (config.method) {
    case DlibAlign: {
        Mat avg_face = imread(faConfig.align_deploy);
        Rect avgfacebbox = Rect(Point(0, 0), avg_face.size());
        alignment_ = new DGFace::DlibAlignment(faConfig.face_size, faConfig.align_model);
        alignment_->set_avgface(avg_face, avgfacebbox);
        break;
    }
    }


    pre_process_ = config.pre_process;
}

FaceFeatureExtractProcessor::~FaceFeatureExtractProcessor() {
    if (recognition_)
        delete recognition_;
    if (alignment_)
        delete alignment_;
    to_processed_.clear();
}
int FaceFeatureExtractProcessor::AlignResult2MatrixAlign(const vector<DGFace::AlignResult> &align_results, vector< Mat > &imgs) {
    for (auto align_result : align_results) {
        imgs.push_back(align_result.face_image);
    }
}

bool FaceFeatureExtractProcessor::process(Frame *frame) {

    int size = frame->objects().size();

    vector<DGFace::AlignResult> align_results;

    for (int i = 0; i < size; ++i) {
        Object *obj = (frame->objects())[i];
        Rect bbox;
        DGFace::AlignResult align_result;
        if (obj && obj->type() == OBJECT_FACE) {
            Face *face = static_cast<Face *>(obj);
            Rect rect;
            rect = face->detection().box;
            Mat img = frame->payload()->data();
            alignment_->align(img, rect, align_result);

        } else {
            DLOG(WARNING) << "Object is not type of face: " << obj->id() << endl;
        }
        align_results.push_back(align_result);

    }
    vector<Mat >align_imgs;
    AlignResult2MatrixAlign(align_results, align_imgs);
    vector<DGFace::RecogResult> results;
    vector<FaceRankFeature> features;
    recognition_->recog(align_imgs, results, pre_process_);
    RecognResult2MatrixRecogn(results, features);

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

    vector<DGFace::AlignResult> align_results;
    for (auto *obj : to_processed_) {
        DGFace::AlignResult align_result;
        Face *face = static_cast<Face *>(obj);
        Mat img = face->image();
        Rect rect = Rect(Point(0, 0), img.size());
        alignment_->align(img, rect, align_result);
        performance_++;
        align_results.push_back(align_result);
    }

    vector<Mat >align_imgs;
    AlignResult2MatrixAlign(align_results, align_imgs);
    vector<FaceRankFeature> features;
    vector<DGFace::RecogResult> results;
    recognition_->recog(align_imgs, results, pre_process_);
    RecognResult2MatrixRecogn(results, features);
    if (features.size() != align_imgs.size()) {
        LOG(ERROR) << "Face image size not equals to feature size: " << align_imgs.size() << ":" << features.size() << endl;
        return false;
    }

    for (int i = 0; i < features.size(); ++i) {
        FaceRankFeature feature = features[i];
        Face *face = (Face *) to_processed_[i];
        face->set_feature(feature);
    }

    return true;
}
int FaceFeatureExtractProcessor::RecognResult2MatrixRecogn(const vector<DGFace::RecogResult> &recog_results, vector< FaceRankFeature > &features) {
    for (auto result : recog_results) {
        FaceRankFeature feature;
        feature.descriptor_ = (result.face_feat);
        features.push_back(feature);
    }
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
