#include "car_rank_processor.h"
#include "processor_helper.h"
namespace dg {
CarRankProcessor::CarRankProcessor()
    : Processor() {

}
CarRankProcessor::~CarRankProcessor() {
}

bool CarRankProcessor::process(Frame *frame) {

    LOG(INFO) << "start process frame: " << frame->id() << endl;

    //process frame
    CarRankFrame *cframe = (CarRankFrame *) frame;
    cframe->result_ = rank(cframe->image_, cframe->hotspots_[0], cframe->candidates_);

    frame->set_status(FRAME_STATUS_FINISHED);
    LOG(INFO) << "end process frame: " << frame->id() << endl;
    return true;
}

vector<Score> CarRankProcessor::rank(const Mat &image, const Rect &hotspot,
                                     const vector<CarRankFeature> &candidates) {
    CarRankFeature des;

    car_feature_extractor_.ExtractDescriptor(image, des);

    LOG(INFO) << "image feature w(" << des.width_ << "), h(" << des.height_ << ")";

    float resize_rto = 600.0 / (float) max(image.cols, image.rows);
    int offset = (600 - resize_rto * image.cols) / 2;

    Rect hotspot_resized(hotspot);
    hotspot_resized.x *= resize_rto;
    hotspot_resized.y *= resize_rto;
    hotspot_resized.width *= resize_rto;
    hotspot_resized.height *= resize_rto;

    LOG(INFO) << "hotspot resized: " << hotspot_resized;

    vector<int> score = car_matcher_.ComputeMatchScore(des, hotspot_resized,
                                                       candidates);


    t_profiler_str_ = "TotalMatching";

    vector<Score> topx(score.size());
    for (int i = 0; i < score.size(); i++) {
        topx[i] = Score(i, score[i]);
    }

    return topx;
}
bool CarRankProcessor::beforeUpdate(FrameBatch *frameBatch) {
#if RELEASE
    performance_=20001;
    if(performance_>20000) {
        if(!RecordFeaturePerformance()) {
            return false;
        }
    }
#endif

    return true;
}
bool CarRankProcessor::RecordFeaturePerformance() {

    return RecordPerformance(FEATURE_CAR_RANK,performance_);

}
}
