#include "car_rank_processor.h"
#include "processor_helper.h"
#include "engine/engine_config_value.h"
#include "util/convert_util.h"

#define MAX_IMAGE_NUM_DEFAULT 10000

using namespace dgvehicle;
namespace dg {
CarRankProcessor::CarRankProcessor()
    : Processor() {
    car_matcher_ = AlgorithmFactory::GetInstance()->CreateCarMatcher();
    car_feature_extractor_ = AlgorithmFactory::GetInstance()->CreateCarFeatureExtractor();
}
CarRankProcessor::~CarRankProcessor() {
    if (car_matcher_)
        delete car_matcher_;
}

bool CarRankProcessor::process(Frame *frame) {

    VLOG(VLOG_RUNTIME_DEBUG) << "Start car ranker process" << frame->id() << endl;

    //process frame
    CarRankFrame *cframe = (CarRankFrame *) frame;
    cframe->result_ = rank(cframe->payload()->data(), cframe->hotspots_[0], cframe->candidates_);

    frame->set_status(FRAME_STATUS_FINISHED);
    return true;
}

vector<Score> CarRankProcessor::rank(const Mat &image, const Rect &hotspot,
                                     const vector<CarRankFeature> &candidates) {
    dgvehicle::CarRankFeature des;

    car_feature_extractor_->ExtractDescriptor(image, des);

    float resize_rto = 600.0 / (float) max(image.cols, image.rows);
    int offset = (600 - resize_rto * image.cols) / 2;

    Rect hotspot_resized(hotspot);
    hotspot_resized.x *= resize_rto;
    hotspot_resized.y *= resize_rto;
    hotspot_resized.width *= resize_rto;
    hotspot_resized.height *= resize_rto;

    vector<dgvehicle::CarRankFeature> vehicleCandidates;
    for (auto candidate : candidates) {
        vehicleCandidates.push_back(ConvertToDgvehicleCarRankFeature(candidate));
    }

    VLOG(VLOG_RUNTIME_DEBUG) << "hotspot resized: " << hotspot_resized;
    vector<int> score = car_matcher_->ComputeMatchScore(des, hotspot_resized, vehicleCandidates);
    vector<Score> topx(score.size());
    for (int i = 0; i < score.size(); i++) {
        topx[i] = Score(i, score[i]);
    }

    return topx;
}
bool CarRankProcessor::beforeUpdate(FrameBatch *frameBatch) {
#if DEBUG
#else
    if (performance_ > RECORD_UNIT) {
        if (!RecordFeaturePerformance()) {
            return false;
        }
    }
#endif

    return true;
}
bool CarRankProcessor::RecordFeaturePerformance() {
    return RecordPerformance(FEATURE_CAR_RANK, performance_);

}
}
