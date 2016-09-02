#include "car_rank_processor.h"
#include "processor_helper.h"
#include "engine/engine_config_value.h"

#define MAX_IMAGE_NUM_DEFAULT 10000

namespace dg {
CarRankProcessor::CarRankProcessor(const Config &config)
    : Processor() {
    int maxImageNum = (int) config.Value(ADVANCED_RANKER_MAXIMUM);
    if (maxImageNum == 0) {
        LOG(ERROR) << "Max image number in car ranker is 0, use default value: " << MAX_IMAGE_NUM_DEFAULT << endl;
        maxImageNum = MAX_IMAGE_NUM_DEFAULT;
    }
    int gpu_id = (int) config.Value(SYSTEM_GPUID);

    car_matcher_ = new CarMatcher(maxImageNum,gpu_id);
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
    CarRankFeature des;

    car_feature_extractor_.ExtractDescriptor(image, des);

    float resize_rto = 600.0 / (float) max(image.cols, image.rows);
    int offset = (600 - resize_rto * image.cols) / 2;

    Rect hotspot_resized(hotspot);
    hotspot_resized.x *= resize_rto;
    hotspot_resized.y *= resize_rto;
    hotspot_resized.width *= resize_rto;
    hotspot_resized.height *= resize_rto;

    VLOG(VLOG_RUNTIME_DEBUG) << "hotspot resized: " << hotspot_resized;
    vector<int> score = car_matcher_->ComputeMatchScore(des, hotspot_resized,
                                                        candidates);
    vector<Score> topx(score.size());
    for (int i = 0; i < score.size(); i++) {
        topx[i] = Score(i, score[i]);
    }

    return topx;
}
bool CarRankProcessor::beforeUpdate(FrameBatch *frameBatch) {
#if DEBUG
#else
    if(performance_>RECORD_UNIT) {
        if(!RecordFeaturePerformance()) {
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
