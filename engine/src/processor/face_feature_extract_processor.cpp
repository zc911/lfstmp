/*============================================================================
 * File Name   : face_feature_extract_processor.cpp
 * Author      : jiajiachen@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年10月21日 下午3:44:11
 * Description :
 * ==========================================================================*/
//#include <alg/feature/face_alignment.h>
#include "processor/face_feature_extract_processor.h"
#include "processor_helper.h"

namespace dg {

FaceFeatureExtractProcessor::FaceFeatureExtractProcessor(
    const FaceFeatureExtractorConfig &config, RecognitionMethod method) {
    method_ = method;
    switch (method_) {
        case RecognitionMethod::CNNRecog:
            LOG(FATAL) << "CNN Recognition not implemented " << endl;
            exit(-1);
//            recognition_ = DGFace::create_recognition(DGFace::recog_method::CNN, config.model_dir,
//                                                      config.gpu_id, false,
//                                                      config.is_model_encrypt, config.batch_size);
            break;
        case RecognitionMethod::LBPRecog: {
            LOG(FATAL) << "LBP Recognition not implemented " << endl;
            exit(-1);
//            recognition_ = DGFace::create_recognition(DGFace::recog_method::LBP, config.model_dir,
//                                                      config.gpu_id, false,
//                                                      config.is_model_encrypt, config.batch_size);
            break;
        }
        case RecognitionMethod::CDNNRecog: {
            LOG(INFO) << "Create Cdnn face recognition " << endl;
            recognition_ = DGFace::create_recognition_with_config(DGFace::recog_method::CDNN, config.model_dir,
                                                                  config.gpu_id, false,
                                                                  config.is_model_encrypt, config.batch_size);
//            recognition_ = new DGFace::CdnnRecog(config.model_config, config.model_dir);
            break;
        }
        case RecognitionMethod::CdnnCaffeRecog: {
            LOG(INFO) << "Create Cdnn caffe face recognition" << endl;
            recognition_ =
                DGFace::create_recognition_with_config(DGFace::recog_method::CDNN_CAFFE, config.model_dir,
                                                       config.gpu_id, false,
                                                       config.is_model_encrypt, config.batch_size);
            break;
        }
        case RecognitionMethod::CdnnFuse: {
            LOG(INFO) << "Create Cdnn fusion face recogniztion" << endl;
            recognition_ = DGFace::create_recognition_with_config(DGFace::recog_method::FUSION, config.model_dir,
                                                                  config.gpu_id, false,
                                                                  config.is_model_encrypt, config.batch_size);
        }

    }

}

FaceFeatureExtractProcessor::~FaceFeatureExtractProcessor() {
    if (recognition_)
        delete recognition_;
    to_processed_.clear();
}

int FaceFeatureExtractProcessor::toAlignmentImages(vector<Mat> &imgs, vector<DGFace::AlignResult> &align_results) {


    if (to_processed_.size() == 0) {
        return -1;
    }

    imgs.clear();
    align_results.clear();

    for (auto itr = to_processed_.begin(); itr != to_processed_.end(); ++itr) {
        Face *face = (Face *) (*itr);
        // no alignment result
        if (face->get_align_result().landmarks.size() == 0) {
            continue;
        }
        align_results.push_back(face->get_align_result());
        imgs.push_back(face->full_image());
    }

    if (imgs.size() == 0 || align_results.size() == 0) {
        LOG(ERROR) << "No alignment results in frame" << endl;
        return -1;
    }

    return 1;

}
static void draw_landmarks(Mat &img, const DGFace::AlignResult &align_result) {
    auto &landmarks = align_result.landmarks;
    for (auto pt = landmarks.begin(); pt != landmarks.end(); ++pt) {
        circle(img, *pt, 2, Scalar(0, 255, 0), -1);
    }
}

bool FaceFeatureExtractProcessor::process(Frame *frame) {

    return true;
}

bool FaceFeatureExtractProcessor::process(FrameBatch *frameBatch) {
    if (to_processed_.size() == 0)
        return true;

    VLOG(VLOG_RUNTIME_DEBUG) << "Start face feature extractor " << endl;

    vector<Mat> align_imgs;
    vector<DGFace::AlignResult> align_results;

    if (toAlignmentImages(align_imgs, align_results) != 1)
        return false;

    vector<FaceRankFeature> features;
    vector<DGFace::RecogResult> results;

    recognition_->recog(align_imgs, align_results, results, "");

    RecognResult2MatrixRecogn(results, features);
    if (features.size() != align_imgs.size()) {
        LOG(ERROR) << "Face image size not equals to feature size: " << align_imgs.size() << ":" << features.size()
            << endl;
        return false;
    }

    for (int i = 0; i < features.size(); ++i) {
        FaceRankFeature feature = features[i];
        Face *face = (Face *) to_processed_[i];
        face->set_feature(feature);
        face->set_align_result(align_results[i]);

    }

    return true;
}

int FaceFeatureExtractProcessor::RecognResult2MatrixRecogn(const vector<DGFace::RecogResult> &recog_results,
                                                           vector<FaceRankFeature> &features) {
    for (auto result : recog_results) {
        FaceRankFeature feature;
        feature.feature_ = (result.face_feat);
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
        } else if (((Face *) (*itr))->image().rows == 0 || ((Face *) (*itr))->image().cols == 0) {
            itr = to_processed_.erase(itr);

        } else {
            itr++;

        }
    }
    //LOG(INFO) << to_processed_.size();
    return true;
}
} /* namespace dg */
