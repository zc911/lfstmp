/*============================================================================
 * File Name   : face_feature_extract_processor.cpp
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年4月21日 下午3:44:11
 * Description :
 * ==========================================================================*/
//#include <alg/feature/face_alignment.h>
#include "processor/face_feature_extract_processor.h"
#include "processor_helper.h"
namespace dg {

FaceFeatureExtractProcessor::FaceFeatureExtractProcessor(
    const FaceFeatureExtractorConfig &config, const FaceAlignmentConfig &faConfig) {
    LOG(INFO) << config.model_config << " " << config.model_dir;
    islog_ = config.islog;
    switch (config.method) {
    case CNNRecog:
        recognition_ = new DGFace::CNNRecog(config.deploy_file, config.model_file, config.layer_name, config.mean, config.pixel_scale, config.use_GPU, config.gpu_id);

        break;
    case LBPRecog: {
        int radius = 1;
        int neighbors = 8;
        int grid_x = 8;
        int grid_y = 8;
        recognition_ = new DGFace::LbpRecog(radius, neighbors, grid_x, grid_y);
        break;
    }
    case CDNNRecog: {
        recognition_ = new DGFace::CdnnRecog(config.model_config, config.model_dir);
        break;
    }
    case CdnnCaffeRecog: {
        recognition_ = new DGFace::CdnnCaffeRecog(config.model_config, config.gpu_id);
        break;
    }
    case CdnnFuse:{
        LOG(INFO)<<config.concurrency;
        recognition_ = new DGFace::FuseRecog(config.model_dir, config.gpu_id,  config.concurrency);

    }

    }
    LOG(INFO) << faConfig.align_model << " " << faConfig.align_path << " " << faConfig.align_cfg << " " << faConfig.align_deploy << faConfig.detect_type;
    switch (config.method) {

    case CDNNRecog: {
        alignment_ = new DGFace::CdnnAlignment(faConfig.face_size, faConfig.align_model);
        align_method_ = CdnnAlign;

        break;

    }
    case CdnnCaffeRecog: {
        LOG(INFO) << faConfig.align_cfg;
        alignment_ = new DGFace::CdnnCaffeAlignment(faConfig.face_size, faConfig.align_path, faConfig.align_cfg, faConfig.gpu_id);
        align_method_ = CdnnCaffeAlign;

        break;
    }
    case CdnnFuse: {
        LOG(INFO) << faConfig.align_cfg;
        alignment_ = new DGFace::CdnnCaffeAlignment(faConfig.face_size, faConfig.align_path, faConfig.align_cfg, faConfig.gpu_id);
           //     alignment_ = new DGFace::CdnnCaffeAlignment(faConfig.face_size, faConfig.align_path, faConfig.align_cfg, faConfig.gpu_id);

        align_method_ = CdnnCaffeAlign;

        break;
    }
    default: {
        Mat avg_face = imread(faConfig.align_deploy);
        Rect avgfacebbox = Rect(Point(0, 0), avg_face.size());
        adjust_box(faConfig.detect_type, avgfacebbox);
        alignment_ = new DGFace::DlibAlignment(faConfig.face_size, faConfig.align_model, faConfig.detect_type);
        alignment_->set_avgface(avg_face, avgfacebbox);
        align_method_ = DlibAlign;

        break;
    }
    }
    face_size_length_ = faConfig.face_size[0];
    align_threshold_ = faConfig.threshold;
    pre_process_ = config.pre_process;
}
void FaceFeatureExtractProcessor::adjust_box(string detect_type, Rect &adjust_box) {
    if (detect_type == "rpn") {

        ///////////////////////////////////////////////////
        // adjust bounding box
        const float h_rate = 0.30;
        const float w_rate = 0.15;

        float a_dist = adjust_box.height * h_rate;

        adjust_box.y += a_dist;
        adjust_box.height -= a_dist;

        a_dist = adjust_box.width * w_rate;
        adjust_box.x += a_dist;
        adjust_box.width -= a_dist * 2;
    } else if (detect_type == "ssd") {
        float a_dist = adjust_box.height * 0.40;

        adjust_box.y += a_dist;
        adjust_box.height -= a_dist;

        a_dist = adjust_box.width * 0.1;
        adjust_box.x += a_dist;
        adjust_box.width -= a_dist * 2;
    }
}

FaceFeatureExtractProcessor::~FaceFeatureExtractProcessor() {
    if (recognition_)
        delete recognition_;
    if (alignment_)
        delete alignment_;
    to_processed_.clear();
}
int FaceFeatureExtractProcessor::AlignResult2MatrixAlign(vector<DGFace::AlignResult> &align_results, vector< Mat > &imgs) {
    if (align_results.size() != to_processed_.size()) {
        return -1;
    }
    vector<Object *>::iterator itr = to_processed_.begin();
    //vector<float>::iterator ditr = det_scores.begin();

    for (vector<DGFace::AlignResult>::iterator aitr = align_results.begin(); aitr != align_results.end();) {
        //    Face *face = static_cast<Face *>(obj);
        //Mat img = face->image();
        float det_threshold  = ((Face *)(*itr))->detection().confidence;
        //LOG(INFO)<<(float)(aitr->score)<<" "<<(float)(((Face *)(*itr))->detection().confidence);
        if (!alignment_->is_face(det_threshold, aitr->score, align_threshold_)) {
            ((Face *)(*itr))->set_valid(false);
            itr = to_processed_.erase(itr);
            aitr = align_results.erase(aitr);
            continue;
        }
        if (align_method_ == DlibAlign) {
            imgs.push_back(aitr->face_image.clone());
        } else {
            bool isValid = true;
            for (auto landmark : aitr->landmarks) {
                if (!landmark.inside(Rect(0, 0, face_size_length_, face_size_length_))) {
                    isValid = false;
                    //   LOG(INFO)<<landmark.x<<" "<<landmark.y;
                    LOG(ERROR) << "landmarks is errors";
                    break;
                }
            }
            if ((aitr->landmarks.size() == 0) || (aitr->face_image.rows == 0) || (aitr->face_image.cols == 0) || (!isValid)) {
                itr = to_processed_.erase(itr);
                aitr = align_results.erase(aitr);

                continue;
            }
            imgs.push_back(aitr->face_image.clone());
        }
        itr++;
        aitr++;
    }
}
static void draw_landmarks(Mat& img, const DGFace::AlignResult &align_result) {
    auto &landmarks = align_result.landmarks;
    for (auto pt = landmarks.begin(); pt != landmarks.end(); ++pt)
    {
        circle(img, *pt, 2, Scalar(0, 255, 0), -1);
    }
}

bool FaceFeatureExtractProcessor::process(Frame *frame) {

    return true;
}

bool FaceFeatureExtractProcessor::process(FrameBatch *frameBatch) {
    if (to_processed_.size() == 0)
        return true;
    vector<DGFace::AlignResult> align_results;
    //vector<float> det_scores;
    for (auto *obj : to_processed_) {
        DGFace::AlignResult align_result;
        Face *face = static_cast<Face *>(obj);
        Mat img = face->image();
        Rect rect = face->detection().box;

        switch (align_method_) {
        case DlibAlign:
            alignment_->align(img, rect, align_result, true);
            break;
        default:
            alignment_->align(img, rect, align_result, false);
            // alignment_->align(img, rect, align_result, false);
            //cout<<"waiting"<<endl;
            break;
        }

        //det_scores.push_back(face->detection().confidence);
        if (islog_) {
            // rectangle(img, rect, Scalar(255, 0, 0));
            // Mat img_draw = align_result.face_image.clone();

            // draw_landmarks(img_draw, align_result);
            // string draw_name = "test_draw" + to_string(performance_) + ".jpg";
            // imwrite(draw_name, img_draw);
            // imwrite("rect.jpg", img);

        }
//LOG(INFO) << "align result box: " << align_result.bbox.x << align_result.bbox.y << " " << align_result.bbox.width << " " << align_result.bbox.height << endl;
//LOG(INFO)  << "align result image: " << align_result.face_image.cols << " " << align_result.face_image.rows << endl;
//LOG(INFO)  << "align result landmarks: " << align_result.landmarks.size() << " " << align_result.landmarks[1].x << " " << align_result.landmarks[1].y << endl;
//LOG(INFO)  << "align result landmarks: " << align_result.landmarks.size() << " " << align_result.landmarks[3].x << " " << align_result.landmarks[3].y << endl;
//LOG(INFO)  << "align result landmarks: " << align_result.landmarks.size() << " " << align_result.landmarks[13].x << " " << align_result.landmarks[13].y << endl;

        performance_++;
        align_results.push_back(align_result);
    }

    vector<Mat >align_imgs;
    AlignResult2MatrixAlign(align_results, align_imgs);
    vector<FaceRankFeature> features;
    vector<DGFace::RecogResult> results;
    recognition_->recog(align_imgs, align_results, results, pre_process_);
    RecognResult2MatrixRecogn(results, features);
    if (features.size() != align_imgs.size()) {
        LOG(ERROR) << "Face image size not equals to feature size: " << align_imgs.size() << ":" << features.size() << endl;

        return false;
    }
    facefilter(frameBatch);
    for (int i = 0; i < features.size(); ++i) {
        FaceRankFeature feature = features[i];
        Face *face = (Face *) to_processed_[i];
        face->set_feature(feature);
    }

    return true;
}
void FaceFeatureExtractProcessor::facefilter(FrameBatch *frameBatch) {
    for (auto *frame : frameBatch->frames()) {
        frame->DeleteInvalidObjects();
    }
}
int FaceFeatureExtractProcessor::RecognResult2MatrixRecogn(const vector<DGFace::RecogResult> &recog_results, vector< FaceRankFeature > &features) {
    for (auto result : recog_results) {
        FaceRankFeature feature;
        feature.descriptor_ = (result.face_feat);
        if (islog_) {
            LOG(INFO)<<result.face_feat.size();
            for (int i = 0; i < result.face_feat.size(); i++) {
                cout << result.face_feat[i] << " ";
            }
            cout << endl;
        }
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
