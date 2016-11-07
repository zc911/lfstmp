#include "rank_engine.h"

#include "processor/face_detect_processor.h"
#include "processor/car_rank_processor.h"
#include "processor/face_rank_processor.h"
#include "processor/config_filter.h"
#include "io/rank_candidates_repo.h"

namespace dg {

SimpleRankEngine::SimpleRankEngine(const Config &config)
    : RankEngine(config),
      id_(0) {

    string type = (string) config.Value(RANKER_DEFAULT_TYPE);
    if (type == "car") {
        enable_ranker_car_ = true;
        enable_ranker_face_ = false;
    } else if (type == "face") {
        enable_ranker_face_ = true;
        enable_ranker_car_ = false;
    } else if ((type == "car|face") || type == "face|car") {
        enable_ranker_car_ = true;
        enable_ranker_face_ = true;
    } else {
        enable_ranker_car_ = true;
        enable_ranker_face_ = false;
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
//        FaceDlibDetector::FaceDetectorConfig fdconfig;
        //configFilter->createFaceDetectorConfig(config, fdconfig);
//        face_detector_ = new FaceDetectProcessor(fdconfig);

//        FaceFeatureExtractor::FaceFeatureExtractorConfig feconfig;
//        configFilter->createFaceExtractorConfig(config, feconfig);
//        face_extractor_ = new FaceFeatureExtractProcessor(feconfig);

        float normalize_alpha = (float) config.Value(ADVANCED_RANKER_NORMALIZE_ALPHA);
        float normalize_beta = (float) config.Value(ADVANCED_RANKER_NORMALIZE_BETA);

        if(normalize_alpha == 0.0f){
            normalize_alpha = -0.05;
        }
        if(normalize_beta == 0.0f){
            normalize_beta = 1.1;
        }

        face_ranker_ = new FaceRankProcessor(normalize_alpha, normalize_beta);
        unsigned int capacity = (int) config.Value(ADVANCED_RANKER_MAXIMUM);
        capacity = capacity <= 0 ? 1 : capacity;
        unsigned int featureLen = (int) config.Value(ADVANCED_RANKER_FEATURE_LENGTH);
        featureLen = featureLen <= 0 ? 256 : featureLen;
        string repoPath = (string) config.Value(ADVANCED_RANKER_REPO_PATH);
        string imageRootPath = (string) config.Value(ADVANCED_RANKER_IMAGE_ROOT_PATH);
        bool needSave = (bool) config.Value(ADVANCED_RANKER_NEED_SAVE_TO_FILE);
        unsigned int saveIterval = (int) config.Value(ADVANCED_RANKER_SAVE_TO_FILE_ITERVAL);
        if(saveIterval < SAVE_TO_FILE_ITERVAL) {
            saveIterval = SAVE_TO_FILE_ITERVAL;
        }

        RankCandidatesRepo::GetInstance().Init(repoPath, imageRootPath, capacity, featureLen, needSave, saveIterval);

    }
}

SimpleRankEngine::~SimpleRankEngine() {
    if (car_ranker_) {
        delete car_ranker_;
    }
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
        face_ranker_->Update(f);
    }
}

void SimpleRankEngine::AddFeatures(FeaturesFrame *f) {
    RankCandidatesRepo &repo = RankCandidatesRepo::GetInstance();
    repo.AddFeatures(*f);
}

}
