//
// Created by chenzhen on 12/2/16.
//

#include "face_alignment_processor.h"
//#include "dgface/alignment/align_dlib.h"
//#include "dgface/alignment/align_cdnn.h"
//#include "dgface/alignment/align_cdnn_caffe.h"
#include "processor_helper.h"
namespace dg {

FaceAlignmentProcessor::FaceAlignmentProcessor(const FaceAlignmentConfig &faConfig,
                                               FaceDetectProcessor::FaceDetectMethod detectMethod) {
    detect_method_ = detectMethod;
    switch (faConfig.method) {


        case CdnnAlign:
            alignment_ = DGFace::create_alignment(DGFace::align_method::CDNN, faConfig.model_dir,
                                                  faConfig.gpu_id, faConfig.is_model_encrypt, faConfig.batch_size);
//            alignment_ = new DGFace::CdnnAlignment(faConfig.face_size, faConfig.align_path);
            align_method_ = CdnnAlign;
            break;
        case DlibAlign: {
            alignment_ = DGFace::create_alignment(DGFace::align_method::DLIB, faConfig.model_dir,
                                                  faConfig.gpu_id, faConfig.is_model_encrypt, faConfig.batch_size);
//            VLOG(VLOG_RUNTIME_DEBUG) << "Use dlib alignment " << endl;
//            Mat avg_face = imread(faConfig.align_deploy);
//            Rect avgfacebbox = Rect(Point(0, 0), avg_face.size());
//            adjust_box(faConfig.detect_type, avgfacebbox);
//            alignment_ = new DGFace::DlibAlignment(faConfig.face_size, faConfig.align_model, faConfig.detect_type);
//            alignment_->set_avgface(avg_face, avgfacebbox);
            align_method_ = DlibAlign;
            break;
        }
        case CdnnCaffeAlign: {
            LOG(ERROR) << "CdnnCaffeAlign method has bug, use cdnn instead currently" << endl;
            alignment_ = DGFace::create_alignment(DGFace::align_method::CDNN_CAFFE, faConfig.model_dir,
                                                  faConfig.gpu_id, faConfig.is_model_encrypt, faConfig.batch_size);
//            alignment_ = new DGFace::CdnnAlignment(faConfig.face_size, faConfig.align_path);
            align_method_ = CdnnAlign;
            break;
        }
        default: {
            LOG(FATAL) << "Face alignment method invalid: " << faConfig.method << endl;
            exit(-1);
        }

    }

    face_size_length_ = faConfig.face_size[0];
    align_threshold_ = faConfig.threshold;
}

FaceAlignmentProcessor::~FaceAlignmentProcessor() {
    if (alignment_)
        delete alignment_;
}

bool FaceAlignmentProcessor::process(Frame *frame) {
    return true;
}

bool FaceAlignmentProcessor::process(FrameBatch *frameBatch) {

    if (to_processed_.size() == 0)
        return true;

    VLOG(VLOG_RUNTIME_DEBUG) << "Start face alignment " << endl;

    for (int i = 0; i < frameBatch->batch_size(); ++i) {
        Frame *f = frameBatch->frames()[i];
    }

    for (auto *obj : to_processed_) {

        DGFace::AlignResult align_result;
        Face *face = static_cast<Face *>(obj);
        Mat img = face->full_image();
        switch (align_method_) {
            case DlibAlign:
                alignment_->align(img, face->detection().rotated_box(), align_result, true);
                break;
            default:
                alignment_->align(img, face->detection().rotated_box(), align_result, false);
                break;
        }

        float det_threshold = face->detection().confidence;
        // only with FCN detection, invoke is_face assert
        if (detect_method_ == FaceDetectProcessor::FcnMethod
            && !alignment_->is_face(det_threshold, align_result.score, align_threshold_)) {
            VLOG(VLOG_RUNTIME_DEBUG) << "Face alignment think this is not a typical face" << endl;
            face->set_valid(false);
            continue;
        }

//        if (align_method_ != DlibAlign) {
//            for (auto landmark : align_result.landmarks) {
//                if (!landmark.inside(Rect(0, 0, face_size_length_, face_size_length_))) {
//                    face->set_valid(false);
//                    LOG(ERROR) << "Face landmarks invalid";
//                    continue;
//                }
//            }
//        }

        face->set_align_result(align_result);
        performance_++;

    }
    frameBatch->FilterInvalid();
    return true;

}

bool FaceAlignmentProcessor::RecordFeaturePerformance() {

    return RecordPerformance(FEATURE_FACE_EXTRACT, performance_);

}


bool FaceAlignmentProcessor::beforeUpdate(FrameBatch *frameBatch) {
#if DEBUG
#else    //#if RELEASE
    if (performance_ > RECORD_UNIT) {
        if (!RecordFeaturePerformance()) {
            return false;
        }
    }
#endif
    to_processed_.clear();
    to_processed_ = frameBatch->CollectObjects(OPERATION_FACE_ALIGNMENT);
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
    return true;
}
} /* namespace dg */

