#include "rank_engine.h"
#include "processor/car_rank_processor.h"
#include "processor/face_rank_processor.h"
namespace dg {

CarRankEngine::CarRankEngine()
        : id_(0) {
    processor_ = new CarRankProcessor();
}
CarRankEngine::~CarRankEngine() {
    if (processor_) {
        delete processor_;
    }
}

vector<Score> CarRankEngine::Rank(const Mat& image, const Rect& hotspot,
                                  const vector<CarRankFeature>& candidates) {
    vector<Rect> hotspots;
    hotspots.push_back(hotspot);
    CarRankFrame f(id_++, image, hotspots, candidates);
    processor_->Update(&f);
    return f.result_;
}

FaceRankEngine::FaceRankEngine()
        : id_(0) {
    processor_ = new FaceRankProcessor();
}

FaceRankEngine::~FaceRankEngine() {
    if (processor_) {
        delete processor_;
    }
}

vector<Score> FaceRankEngine::Rank(const Mat& image, const Rect& hotspot,
                                   const vector<FaceRankFeature>& candidates) {
    vector<Rect> hotspots;
    hotspots.push_back(hotspot);
    FaceRankFrame f(id_++, image, hotspots, candidates);
    processor_->Update(&f);
    return f.result_;
}

}
