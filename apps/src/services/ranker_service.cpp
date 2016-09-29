/*============================================================================
 * File Name   : ranker_service.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#include <glog/logging.h>
#include "ranker_service.h"
#include "codec/base64.h"
#include "image_service.h"
#include "../../../engine/src/io/rank_candidates_repo.h"

namespace dg {
//const int RANKER_MAXIMUM = 10000
RankerAppsService::RankerAppsService(const Config *config, string name, int baseId)
    : name_(name),
      config_(config) {
    config_ = config;

    limits_ = min(RANKER_MAXIMUM, (int) config->Value(ADVANCED_RANKER_MAXIMUM));

}

RankerAppsService::~RankerAppsService() {

}

MatrixError RankerAppsService::RankFeature(const RankFeatureRequest *request, RankFeatureResponse *response) {

    MatrixError err;

    switch (request->context().type()) {
        case dg::RANK_TYPE_FACE:
            return getRankedFaceVector(request, response);
        default:
            LOG(ERROR) << "bad request(" << request->context().sessionid() << "), unknown rank type "
                << request->context().type();
            err.set_code(-1);
            err.set_message("bad request, unknown action");
            return err;
    }

    return err;

}

//
//MatrixError RankerAppsService::GetRankedVector(
//    const FeatureRankingRequest *request,
//    FeatureRankingResponse *response) {
//    MatrixError err;
//    try {
//
//        switch (request->type()) {
//            case dg::REC_TYPE_VEHICLE:
//                return getRankedCarVector(request, response);
//            case dg::REC_TYPE_FACE:
//                return getRankedFaceVector(request, response);
//            case dg::REC_TYPE_ALL:
//                return getRankedAllVector(request, response);
//            case dg::REC_TYPE_DEFAULT:
//                return getRankedCarVector(request, response);
//
//            default:
//                LOG(ERROR) << "bad request(" << request->reqid() << "), unknown action";
//                err.set_code(-1);
//                err.set_message("bad request, unknown action");
//                return err;
//        }
//    }
//    catch (const std::exception &e) {
//        LOG(WARNING) << "bad request(" << request->reqid() << "), " << e.what() << endl;
//        err.set_code(-1);
//        err.set_message(e.what());
//        return err;
//    }
//}
//MatrixError RankerAppsService::getRankedAllVector(
//    const FeatureRankingRequest *request,
//    FeatureRankingResponse *response) {
//    vector<Score> faceScores;
//    vector<Score> carScores;
//    MatrixError err;
//    err = getFaceScoredVector(faceScores, request, response);
//    if (err.code() != 0) {
//        return err;
//    }
//
//    err = getCarScoredVector(carScores, request, response);
//    if (err.code() != 0) {
//        return err;
//    }
//
//    int limit = getLimit(request);
//    if (faceScores.size() > 0) {
//        partial_sort(faceScores.begin(), faceScores.begin() + limit,
//                     faceScores.end());
//        faceScores.resize(limit);
//    }
//    if (carScores.size() > 0) {
//
//        partial_sort(carScores.begin(), carScores.begin() + limit,
//                     carScores.end());
//        carScores.resize(limit);
//    }
//    for (Score &s : faceScores) {
//        response->add_ids(request->candidates(s.index_).id());
//        response->add_scores(s.score_);
//    }
//    for (Score &s : carScores) {
//        response->add_ids(request->candidates(s.index_).id());
//        response->add_scores(s.score_);
//    }
//}
//MatrixError RankerAppsService::getRankedCarVector(
//    const FeatureRankingRequest *request,
//    FeatureRankingResponse *response) {
//    vector<Score> scores;
//    MatrixError err;
//
//    err = getCarScoredVector(scores, request, response);
//
//    //sort & fill
//    sortAndFillResponse(request, scores, response);
//    return err;
//}
//MatrixError RankerAppsService::getCarScoredVector(
//    vector<Score> &scores, const FeatureRankingRequest *request,
//    FeatureRankingResponse *response) {
//    string prefix = requestPrefix(request);
//    response->set_reqid(request->reqid());
//
//    MatrixError err;
//
//    if (!request->has_image()) {
//        LOG(ERROR) << prefix << "image descriptor does not exist";
//        err.set_code(-1);
//        err.set_message("image descriptor does not exist");
//        return err;
//    }
//
//    Mat image;
//    err = ImageService::ParseImage(request->image(), image);
//    if (err.code() != 0) {
//        LOG(ERROR) << prefix << "parse image failed, " << err.message();
//        return err;
//    }
//
//    Rect hotspot = getHotspot(request, image);
//
//    vector<CarRankFeature> features;
//    err = extractFeatures(request, features, limits_);
//
//    if (err.code() != 0) {
//        LOG(ERROR) << prefix << "parse candidates failed, " << err.message();
//        return err;
//    }
//    vector<Rect> hotspots;
//    hotspots.push_back(hotspot);
//    CarRankFrame f(0, image, hotspots, features);
//
//    MatrixEnginesPool<SimpleRankEngine> *engine_pool = MatrixEnginesPool<SimpleRankEngine>::GetInstance();
//
//    EngineData data;
//    data.func = [&f, &data]() -> void {
//      return (bind(&SimpleRankEngine::RankCar, (SimpleRankEngine *) data.apps,
//                   placeholders::_1))(&f);
//    };
//
//    if (engine_pool == NULL) {
//        LOG(ERROR) << "Engine pool not initailized. " << endl;
//        return err;
//    }
//    engine_pool->enqueue(&data);
//    data.Wait();
//
//    scores = f.result_;
//
//    //scores = car_ranker_.Rank(image, hotspot, features);
//    return err;
//}

MatrixError RankerAppsService::getFaceScoredVector(
    vector<Score> &scores, const RankFeatureRequest *request,
    RankFeatureResponse *response) {


    response->mutable_context()->set_sessionid(request->context().sessionid());

    MatrixError err;

    if (request->feature().feature().size() == 0) {
        LOG(ERROR) << "Compared feature vector is empty" << endl;
        err.set_code(-1);
        err.set_message("Compared feature vector is empty");
        return err;
    }


    FaceRankFeature feature;
    Base64::Decode(request->feature().feature(), feature.descriptor_);
    cout << "input feature size: " << feature.descriptor_.size() << endl;

    FaceRankFrame f(0, feature);
    Operation op;

    op.Set(OPERATION_FACE_FEATURE_VECTOR);

    f.set_operation(op);
    MatrixEnginesPool<SimpleRankEngine> *engine_pool = MatrixEnginesPool<SimpleRankEngine>::GetInstance();
    EngineData data;
    data.func = [&f, &data]() -> void {
        return (bind(&SimpleRankEngine::RankFace, (SimpleRankEngine *) data.apps,
                     placeholders::_1))(&f);
    };

    if (engine_pool == NULL) {
        LOG(ERROR) << "Engine pool not initailized. " << endl;
        return err;
    }

    engine_pool->enqueue(&data);
    data.Wait();

    RankCandidatesRepo &repo = RankCandidatesRepo::GetInstance();
    for (auto r : f.result_) {
        RankResult *result = response->mutable_candidates()->Add();
        const RankCandidatesItem &item = repo.Get(r.index_);
        result->set_uri(item.image_uri);
        result->set_id(r.index_);
        result->set_score(1.0);
        vector<uchar> data(item.image.datastart, item.image.dataend);
        result->set_data(Base64::Encode<uchar>(data));
    }


    return err;
}

MatrixError RankerAppsService::getRankedFaceVector(
    const RankFeatureRequest *request,
    RankFeatureResponse *response) {

    MatrixError err;

    vector<Score> scores;
    err = getFaceScoredVector(scores, request, response);

    return err;
}

//void RankerAppsService::sortAndFillResponse(
//    const FeatureRankingRequest *request, vector<Score> &scores,
//    FeatureRankingResponse *response) {
//    if (scores.size() == 0) {
//        return;
//    }
//    int limit = getLimit(request);
//
//    partial_sort(scores.begin(), scores.begin() + limit, scores.end());
//    scores.resize(limit);
//    for (Score &s : scores) {
//        response->add_ids(request->candidates(s.index_).id());
//        response->add_scores(s.score_);
//    }
//}
//
//string RankerAppsService::requestPrefix(const FeatureRankingRequest *request) {
//    stringstream ss;
//    ss << "request(" << request->reqid() << "): ";
//    return ss.str();
//}
//
//Rect RankerAppsService::getHotspot(const FeatureRankingRequest *request,
//                                   const Mat &image) {
//    if (request->interestedareas_size() > 0) {
//        const Cutboard &cb = request->interestedareas(0);
//        if (cb.width() != 0 && cb.height() != 0) {
//            return Rect(cb.x(), cb.y(), cb.width(), cb.height());
//        }
//    }
//    return Rect(0, 0, image.cols, image.rows);
//}

//int RankerAppsService::getLimit(const FeatureRankingRequest *request) {
//    int limit = request->limit();
//    if (limit <= 0 || limit >= request->candidates_size()) {
//        limit = request->candidates_size();
//    }
//    return limit;
//}

}
