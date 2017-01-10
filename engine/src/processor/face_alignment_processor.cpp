//
// Created by chenzhen on 12/2/16.
//

#include "face_alignment_processor.h"
#include "processor_helper.h"

namespace dg {

FaceAlignmentProcessor::FaceAlignmentProcessor(const FaceAlignmentConfig &faConfig,
                                               FaceAlignmentProcessor::AlignmentMethod alignMethod,
                                               FaceDetectProcessor::DetectMethod detectMethod) {
    detect_method_ = detectMethod;
    align_method_ = alignMethod;
    switch (align_method_) {
        case AlignmentMethod::CdnnAlign:
            alignment_ = DGFace::create_alignment_with_global_dir(DGFace::align_method::CDNN,
                                                                  faConfig.model_dir,
                                                                  faConfig.gpu_id,
                                                                  faConfig.is_model_encrypt,
                                                                  faConfig.batch_size);
            break;
        case AlignmentMethod::DlibAlign: {
            LOG(FATAL) << "Dlib method not implemented, use cdnn instead currently" << endl;
            exit(-1);
            break;
        }
        case AlignmentMethod::CdnnCaffeAlign: {
            LOG(FATAL) << "CdnnCaffeAlign method has bug, use cdnn instead currently" << endl;
            exit(-1);
            break;
        }
        default: {
            LOG(FATAL) << "Face alignment method invalid: " << (int) detectMethod << endl;
            exit(-1);
        }

    }

    align_threshold_ = faConfig.align_threshold;
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
            case AlignmentMethod::DlibAlign:
                alignment_->align(img, face->detection().rotated_box(), align_result, true);
                break;
            default:
                alignment_->align(img, face->detection().rotated_box(), align_result, false);
                break;
        }

        float det_threshold = face->detection().confidence;
        // only with FCN detection, invoke is_face assert
        if (detect_method_ == FaceDetectProcessor::DetectMethod::FcnMethod
            && !alignment_->is_face(det_threshold, align_result.score, align_threshold_)) {
            VLOG(VLOG_RUNTIME_DEBUG) << "Face alignment think this is not a typical face" << endl;
            face->set_valid(false);
            continue;
        }

        face->set_align_result(align_result);
        performance_++;

    }
    frameBatch->FilterInvalid();
    return true;

}

bool FaceAlignmentProcessor::RecordFeaturePerformance() {
    return true;
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
        } else if (((Face * )(*itr))->image().rows == 0 || ((Face * )(*itr))->image().cols == 0) {
            itr = to_processed_.erase(itr);
        } else {
            itr++;

        }

    }
    return true;
}
} /* namespace dg */

