#include "rank_engine.h"

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

FaceRankEngine::FaceRankEngine(const Config &config)
        : id_(0) {
    init(config);
}
void FaceRankEngine::init(const Config &config) {

    ConfigFilter *configFilter = ConfigFilter::GetInstance();
    if (!configFilter->initDataConfig(config)) {
        LOG(ERROR)<<"can not init data config"<<endl;
        DLOG(ERROR)<<"can not init data config"<<endl;
        return;
    }
    FaceDetector::FaceDetectorConfig fdconfig;
    configFilter->createFaceDetectorConfig(config, fdconfig);
    detector_ = new FaceDetectProcessor(fdconfig);

    FaceFeatureExtractor::FaceFeatureExtractorConfig feconfig;
    configFilter->createFaceExtractorConfig(config, feconfig);
    extractor_ = new FaceFeatureExtractProcessor(feconfig);

    ranker_ = new FaceRankProcessor();

}
FaceRankEngine::~FaceRankEngine() {
    delete detector_;
    delete extractor_;
    delete ranker_;
}

vector<Score> FaceRankEngine::Rank(const Mat& image, const Rect& hotspot,
                                   const vector<FaceRankFeature>& candidates) {

    Frame *frame = new Frame(0, image);
    detector_->Update(frame);
    extractor_->Update(frame);

    Face *face = (Face *) frame->get_object(0);
    FaceRankFeature feature = face->feature();
    vector<Rect> hotspots;
    hotspots.push_back(hotspot);

    FaceRankFrame *face_rank_frame = new FaceRankFrame(0, feature, hotspots,
                                                       candidates);
    ranker_->Update(face_rank_frame);

    vector<Score> result = face_rank_frame->result_;
    delete frame;
    delete face_rank_frame;

    return result;
}

}
