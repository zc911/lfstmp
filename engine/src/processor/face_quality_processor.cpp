#include "processor/face_quality_processor.h"

#include "processor_helper.h"
namespace dg {

FaceQualityProcessor::FaceQualityProcessor(const FaceQualityConfig &config) {
    blur_quality_ = DGFace::create_quality(DGFace::BLURM, "", 0);
    pose_quality_ = DGFace::create_quality(DGFace::POSE, "", 0);

    blur_threshold_ = config.blur_threshold;
}

FaceQualityProcessor::~FaceQualityProcessor() {

    delete blur_quality_;
    delete pose_quality_;
}

bool FaceQualityProcessor::process(FrameBatch *frameBatch) {

    if (!(blur_quality_ && pose_quality_)) {
        LOG(ERROR) << "Quality processor init error, skip" << endl;
        return false;
    }

    VLOG(VLOG_RUNTIME_DEBUG) << "Start face quality filter " << endl;

    for (vector<Object *>::iterator itr = to_processed_.begin(); itr != to_processed_.end(); itr++) {

        Face *face = (Face * )(*itr);
        Mat faceImg = face->image();


        // apply blur quality filter
        float blur_score = blur_quality_->quality(faceImg);
        if (blur_threshold_ > blur_score) {
            VLOG(VLOG_RUNTIME_DEBUG) << "Blur filter failed, score is " << blur_score << " and threshold is "
                << blur_threshold_ << endl;
            face->set_valid(false);
            continue;
        }
        face->set_qualities(Face::BlurM, blur_score);

        // get face pose
        if(face->get_align_result().landmarks.size() != 0){
            vector<float> scores = pose_quality_->quality(face->get_align_result());
            face->set_pose(scores);
        }else{
            LOG(ERROR) << "There is not align result for face " << face->id() << endl;
        }


        performance_++;
    }

    frameBatch->FilterInvalid();

    return true;
}

bool FaceQualityProcessor::beforeUpdate(FrameBatch *frameBatch) {
#if DEBUG
#else
    if (performance_ > RECORD_UNIT) {
        if (!RecordFeaturePerformance()) {
            return false;
        }
    }
#endif
    to_processed_.clear();
    to_processed_ = frameBatch->CollectObjects(OPERATION_FACE_QUALITY);
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
bool FaceQualityProcessor::RecordFeaturePerformance() {
    return true;
}
}