#include "rank_engine.h"

#include "processor/face_detect_processor.h"
#include "processor/face_feature_extract_processor.h"
#include "processor/car_rank_processor.h"
#include "processor/face_rank_processor.h"
#include "processor/config_filter.h"

namespace dg {

SimpleRankEngine::SimpleRankEngine(const Config &config)
    : RankEngine(config),
      id_(0) {
#if DEBUG
    enable_ranker_face_ = true;
    enable_ranker_vehicle_ = true;
#else
    enable_ranker_vehicle_ = (CheckFeature(FEATURE_CAR_RANK, false) == ERR_FEATURE_ON);
    enable_ranker_face_ = (CheckFeature(FEATURE_FACE_RANK, false) == ERR_FEATURE_ON) && (CheckFeature(FEATURE_FACE_EXTRACT, false) == ERR_FEATURE_ON) && (CheckFeature(FEATURE_FACE_DETECTION, false) == ERR_FEATURE_ON);

#endif

    if (enable_ranker_vehicle_) {
        processor_ = new CarRankProcessor(config);
    }
    if (enable_ranker_face_) {
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

SimpleRankEngine::~SimpleRankEngine() {
    if (processor_) {
        delete processor_;
    }
}

void SimpleRankEngine::Rank(RankFrame *f) {
    if (processor_) {
        vector<Rect> hotspots;
        hotspots.push_back(hotspot);
        f->set_id(id_++);
        processor_->Update(f);
    } else {
    }

}

FaceRankEngine::FaceRankEngine(const Config &config)
    : RankEngine(config),
      id_(0) {
    init(config);
}
void FaceRankEngine::init(const Config &config) {
#if DEBUG
    enable_ranker_ = true;
#else
    enable_ranker_ = (CheckFeature(FEATURE_FACE_RANK, false) == ERR_FEATURE_ON) && (CheckFeature(FEATURE_FACE_EXTRACT, false) == ERR_FEATURE_ON) && (CheckFeature(FEATURE_FACE_DETECTION, false) == ERR_FEATURE_ON);
#endif
    enable_ranker_ = true;

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

void FaceRankEngine::Rank(FaceRankFrame *face_rank_frame) {
    if (!enable_ranker_)
        return;
    detector_->Update(face_rank_frame);
    extractor_->Update(face_rank_frame);

    Face *face = (Face *) face_rank_frame->get_object(0);
    if (face != NULL) {

        FaceRankFeature feature = face->feature();
        face_rank_frame->set_feature(feature);

        ranker_->Update(face_rank_frame);
    }
}

}
