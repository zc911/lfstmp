#include "car_rank_processor.h"
namespace dg {
CarRankProcessor::CarRankProcessor()
        : Processor() {

}
CarRankProcessor::~CarRankProcessor() {
}

void CarRankProcessor::Update(Frame *frame) {

    LOG(INFO)<< "start process frame: " << frame->id() << endl;

    //process frame
    CarRankFrame *cframe = (CarRankFrame *)frame;
    cframe->result_ = rank(cframe->image_, cframe->hotspots_[0], cframe->candidates_);

    frame->set_status(FRAME_STATUS_FINISHED);
    LOG(INFO) << "end process frame: " << frame->id() << endl;
}

void CarRankProcessor::Update(FrameBatch *frameBatch) {

}

bool CarRankProcessor::checkOperation(Frame *frame) {
    return true;
}

bool CarRankProcessor::checkStatus(Frame *frame) {
    return frame->status() == FRAME_STATUS_FINISHED ? false : true;
}

vector<Score> CarRankProcessor::rank(const Mat& image, const Rect& hotspot,
                                     const vector<CarRankFeature>& candidates) {
    CarRankFeature des;
    car_feature_extractor_.ExtractDescriptor(image, des);
    LOG(INFO)<< "image feature w(" << des.width_ << "), h(" << des.height_ << ")";

    float resize_rto = 600.0 / (float) max(image.cols, image.rows);
    int offset = (600 - resize_rto * image.cols) / 2;

    Rect hotspot_resized(hotspot);
    hotspot_resized.x *= resize_rto;
    hotspot_resized.y *= resize_rto;
    hotspot_resized.width *= resize_rto;
    hotspot_resized.height *= resize_rto;

//        hotspot_resized.x = 1.0 * (hotspot_resized.x - offset) / resize_rto;
//        hotspot_resized.y = 1.0 * (hotspot_resized.y - offset) / resize_rto;
//        hotspot_resized.width = 1.0 * hotspot_resized.width / resize_rto;
//        hotspot_resized.height = 1.0 * hotspot_resized.height / resize_rto;
    LOG(INFO)<< "hotspot resized: " << hotspot_resized;

    t_profiler_matching_.Reset();
    vector<int> score = car_matcher_.ComputeMatchScore(des, hotspot_resized,
                                                       candidates);
    t_profiler_str_ = "TotalMatching";
    t_profiler_matching_.Update(t_profiler_str_);

    vector<Score> topx(score.size());
    for (int i = 0; i < score.size(); i++) {
        topx[i] = Score(i, score[i]);
    }

    LOG(INFO)<< "Ranking finished, " <<t_profiler_matching_.getSmoothedTimeProfileString();
    return topx;
}
}
