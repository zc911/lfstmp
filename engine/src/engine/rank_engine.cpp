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

    int type = (int) config.Value(RANKER_DEFAULT_TYPE);
    switch (type) {
        case 0:
        case 1:
            enable_ranker_car_ = true;
            enable_ranker_face_ = false;
            break;
        case 2:
            enable_ranker_face_ = true;
            enable_ranker_car_ = false;

            break;
        case 3:
            enable_ranker_car_ = true;
            enable_ranker_face_ = true;
            break;
        default:
            enable_ranker_face_ = false;
            enable_ranker_car_ = false;

            break;
    }
#if DEBUG
#else
    enable_ranker_car_ = enable_ranker_car_ && (CheckFeature(FEATURE_CAR_RANK, false) == ERR_FEATURE_ON);
    enable_ranker_face_ = enable_ranker_face_ && (CheckFeature(FEATURE_FACE_RANK, false) == ERR_FEATURE_ON)
        && (CheckFeature(FEATURE_FACE_EXTRACT, false) == ERR_FEATURE_ON)
        && (CheckFeature(FEATURE_FACE_DETECTION, false) == ERR_FEATURE_ON);

#endif

    if (enable_ranker_car_) {
        car_ranker_ = new CarRankProcessor(config);
    }
    if (enable_ranker_face_) {
        ConfigFilter *configFilter = ConfigFilter::GetInstance();
        if (!configFilter->initDataConfig(config)) {
            LOG(ERROR) << "can not init data config" << endl;
            DLOG(ERROR) << "can not init data config" << endl;
            return;
        }
//        FaceDetector::FaceDetectorConfig fdconfig;
//        configFilter->createFaceDetectorConfig(config, fdconfig);
//        face_detector_ = new FaceDetectProcessor(fdconfig);

//        FaceFeatureExtractor::FaceFeatureExtractorConfig feconfig;
//        configFilter->createFaceExtractorConfig(config, feconfig);
//        face_extractor_ = new FaceFeatureExtractProcessor(feconfig);

        face_ranker_ = new FaceRankProcessor();

    }
}

SimpleRankEngine::~SimpleRankEngine() {
    if (car_ranker_) {
        delete car_ranker_;
    }
//    if (face_detector_) {
//        delete face_detector_;
//    }
//    if (face_extractor_) {
//        delete face_extractor_;
//    }
    if (face_ranker_) {
        delete face_ranker_;
    }
}

void SimpleRankEngine::RankCar(CarRankFrame *f) {
    if (enable_ranker_car_) {
        f->set_id(id_++);
        car_ranker_->Update(f);
    }
}
void SimpleRankEngine::RankFace(FaceRankFrame *f) {

    if (enable_ranker_face_) {
//        face_detector_->Update(f);
//        face_extractor_->Update(f);

//        Face *face = (Face *) f->get_object(0);
//        if (face != NULL) {

//        FaceRankFeature feature = face->feature();
//        f->set_feature(feature);

        face_ranker_->Update(f);
//        }
    }
}

}
