#include "face_rank_processor.h"

namespace dg {

FaceRankProcessor::FaceRankProcessor()
        : Processor() {
    extractor_ = new FaceFeatureExtractor("models/deployface1",
                                          "models/modelface1", true, 1, "models/shapeface1", "models/avgface1");
}
FaceRankProcessor::~FaceRankProcessor() {
}

void FaceRankProcessor::Update(Frame *frame) {
    if (!checkOperation(frame)) {
        LOG(INFO)<< "operation no allowed" << endl;
        return;
    }
    if (!checkStatus(frame)) {
        LOG(INFO) << "check status failed " << endl;
        return;
    }
    LOG(INFO) << "start process frame: " << frame->id() << endl;

    //process frame
    FaceRankFrame *fframe = (FaceRankFrame *)frame;
    fframe->result_ = rank(fframe->image_, fframe->hotspots_[0], fframe->candidates_);

    frame->set_status(FRAME_STATUS_FINISHED);
    LOG(INFO) << "end process frame: " << frame->id() << endl;
}

void FaceRankProcessor::Update(FrameBatch *frameBatch) {

}

bool FaceRankProcessor::checkOperation(Frame *frame) {
    return true;
}

bool FaceRankProcessor::checkStatus(Frame *frame) {
    return frame->status() == FRAME_STATUS_FINISHED ? false : true;
}

vector<Score> FaceRankProcessor::rank(
        const Mat& image, const Rect& hotspot,
        const vector<FaceRankFeature>& candidates) {
    std::vector<Mat> images;
    images.push_back(image);
    std::vector<FaceFeature> features = extractor_->Extract(images);

    vector<float> feature(features[0].data, features[0].data + 256);

    vector<Score> pred;
    for (int i = 0; i < features.size(); i++) {
        Score p(i, getCosSimilarity(feature, candidates[i].descriptor_));
        pred.push_back(p);
    }
    return pred;
}

float FaceRankProcessor::getCosSimilarity(const vector<float> & A,
                                          const vector<float> & B) {
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (unsigned int i = 0; i < A.size(); ++i) {
        dot += A[i] * B[i];
        denom_a += A[i] * A[i];
        denom_b += B[i] * B[i];
    }
    return abs(dot) / (sqrt(denom_a) * sqrt(denom_b));
}

}
