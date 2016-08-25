#include "rank_engine.h"

#include "processor/face_detect_processor.h"
#include "processor/face_feature_extract_processor.h"
#include "processor/car_rank_processor.h"
#include "processor/face_rank_processor.h"
#include "processor/config_filter.h"

namespace dg {

CarRankEngine::CarRankEngine(const Config &config)
    : RankEngine(config),
      id_(0) {
#if DEBUG
    enable_ranker_=true;    
#else
    enable_ranker_ = (CheckFeature(FEATURE_CAR_RANK, false) == ERR_FEATURE_ON);
#endif
        enable_ranker_=true;    

    if (enable_ranker_) {
        processor_ = new CarRankProcessor(config);
    }
}

CarRankEngine::~CarRankEngine() {
    if (processor_) {
        delete processor_;
    }
}

vector<Score> CarRankEngine::Rank(const Mat &image, const Rect &hotspot,
                                  const vector<CarRankFeature> &candidates) {
    if (processor_) {
        vector<Rect> hotspots;
        hotspots.push_back(hotspot);
        CarRankFrame f(id_++, image, hotspots, candidates);
        processor_->Update(&f);
        return f.result_;
    } else {
        return vector<Score>();
    }

}

FaceRankEngine::FaceRankEngine(const Config &config)
    : RankEngine(config),
      id_(0) {
    init(config);
}
void FaceRankEngine::init(const Config &config) {
#if DEBUG
    enable_ranker_=true;    
#else
    enable_ranker_ = (CheckFeature(FEATURE_FACE_RANK, false) == ERR_FEATURE_ON) && (CheckFeature(FEATURE_FACE_EXTRACT, false) == ERR_FEATURE_ON) && (CheckFeature(FEATURE_FACE_DETECTION, false) == ERR_FEATURE_ON);
#endif
        enable_ranker_=true;    

    if (enable_ranker_) {
        ConfigFilter *configFilter = ConfigFilter::GetInstance();
        if (!configFilter->initDataConfig(config)) {
            LOG(ERROR) << "can not init data config" << endl;
            DLOG(ERROR) << "can not init data config" << endl;
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

}

FaceRankEngine::~FaceRankEngine() {
    delete detector_;
    delete extractor_;
    delete ranker_;
}

vector<Score> FaceRankEngine::Rank(const Mat &image, const Rect &hotspot,
                                   const vector<FaceRankFeature> &candidates) {
    vector<Score> result;
    if (!enable_ranker_)
        return result;

    Frame *frame = new Frame(0, image);

    Operation op;

    op.Set(OPERATION_FACE | OPERATION_FACE_DETECTOR
           | OPERATION_FACE_FEATURE_VECTOR);

    frame->set_operation(op);

    detector_->Update(frame);
    extractor_->Update(frame);

    Face *face = (Face *) frame->get_object(0);
    if (face != NULL) {

        FaceRankFeature feature = face->feature();
        vector<Rect> hotspots;
        hotspots.push_back(hotspot);

        FaceRankFrame *face_rank_frame = new FaceRankFrame(0, feature, hotspots,
                candidates);

        ranker_->Update(face_rank_frame);

        result = face_rank_frame->result_;
        delete face_rank_frame;

    }
    delete frame;
    return result;
}

}
