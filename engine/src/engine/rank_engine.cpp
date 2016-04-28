#include "rank_engine.h"
#include "model/frame.h"
namespace dg {

CarRankEngine::CarRankEngine()
        : id_(0) {
}
CarRankEngine::~CarRankEngine() {
}

vector<Score> CarRankEngine::Rank(const Mat& image, const Rect& hotspot,
                                  const vector<CarRankFeature>& candidates) {
    vector<Rect> hotspots;
    hotspots.push_back(hotspot);
    CarRankFrame f(id_++, image, hotspots, candidates);
    processor_.Update(&f);
    return f.result_;
}

FaceRankEngine::FaceRankEngine()
        : id_(0) {
}

FaceRankEngine::~FaceRankEngine() {
}

vector<Score> FaceRankEngine::Rank(const Mat& image, const Rect& hotspot,
                                   const vector<FaceRankFeature>& candidates) {
    vector<Rect> hotspots;
    hotspots.push_back(hotspot);
    FaceRankFrame f(id_++, image, hotspots, candidates);
    processor_.Update(&f);
    return f.result_;
}
}
