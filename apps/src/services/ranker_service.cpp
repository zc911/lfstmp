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

namespace dg {

RankerAppsService::RankerAppsService(const Config *config)
    : config_(config),
      car_ranker_(*config),
      face_ranker_(*config) {

    int limits=car_ranker_.GetMaxCandidatesSize();
    LOG(INFO)<<"Limits "<<limits<<endl;
}

RankerAppsService::~RankerAppsService() {

}

MatrixError RankerAppsService::GetRankedVector(const FeatureRankingRequest *request,
                                               FeatureRankingResponse *response) {
    MatrixError err;
    try {

        switch (request->type()) {
            case dg::REC_TYPE_VEHICLE:
                return getRankedCarVector(request, response);
            case dg::REC_TYPE_FACE:
                return getRankedFaceVector(request, response);
            case dg::REC_TYPE_ALL:
                LOG(WARNING) << "bad request(" << request->reqid() << "), all type not supported";
                err.set_code(-1);
                err.set_message("bad request all type not supported");
                return err;
            case dg::REC_TYPE_DEFAULT:
            default:
                LOG(ERROR) << "bad request(" << request->reqid() << "), unknown action";
                err.set_code(-1);
                err.set_message("bad request, unknown action");
                return err;
        }
    }
    catch (const std::exception &e) {
        LOG(WARNING) << "bad request(" << request->reqid() << "), " << e.what() << endl;
        err.set_code(-1);
        err.set_message(e.what());
        return err;
    }
}

MatrixError RankerAppsService::getRankedCarVector(const FeatureRankingRequest *request,
                                                  FeatureRankingResponse *response) {
    string prefix = requestPrefix(request);
    LOG(INFO) << prefix << "started";
    response->set_reqid(request->reqid());

    MatrixError err;

    if (!request->has_image()) {
        LOG(ERROR) << prefix << "image descriptor does not exist";
        err.set_code(-1);
        err.set_message("image descriptor does not exist");
        return err;
    }

    Mat image;
    err = ImageService::ParseImage(request->image(), image);
    if (err.code() != 0) {
        LOG(ERROR) << prefix << "parse image failed, " << err.message();
        return err;
    }

    Rect hotspot = getHotspot(request, image);

    int limits=car_ranker_.GetMaxCandidatesSize();
    vector<CarRankFeature> features;
    err = extractFeatures(request, features,limits);

    if (err.code() != 0) {
        LOG(ERROR) << prefix << "parse candidates failed, " << err.message();
        return err;
    }

    vector<Score> scores = car_ranker_.Rank(image, hotspot, features);

    //sort & fill
    sortAndFillResponse(request, scores, response);
    return err;
}
MatrixError RankerAppsService::getRankedFaceVector(
    const FeatureRankingRequest *request,
    FeatureRankingResponse *response) {
    string prefix = requestPrefix(request);
    LOG(INFO) << prefix << "started";
    response->set_reqid(request->reqid());

    MatrixError err;
    if (!request->has_image()) {
        LOG(ERROR) << prefix << "image descriptor does not exist";
        err.set_code(-1);
        err.set_message("image descriptor does not exist");
        return err;
    }

    Mat image;
    err = ImageService::ParseImage(request->image(), image);
    if (err.code() != 0) {
        LOG(ERROR) << prefix << "parse image failed, " << err.message();
        return err;
    }

    Rect hotspot = getHotspot(request, image);

    int limits=car_ranker_.GetMaxCandidatesSize();
    vector<FaceRankFeature> features;
    err = extractFeatures(request, features,limits);
    if (err.code() != 0) {
        LOG(ERROR) << prefix << "parse candidates failed, " << err.message();
        return err;
    }

    vector<Score> scores = face_ranker_.Rank(image, hotspot, features);

    //sort & fill
    sortAndFillResponse(request, scores, response);
    return err;
}

void RankerAppsService::sortAndFillResponse(
    const FeatureRankingRequest *request, vector<Score> &scores,
    FeatureRankingResponse *response) {
    int limit = getLimit(request);

    partial_sort(scores.begin(), scores.begin() + limit, scores.end());
    scores.resize(limit);

    for (Score &s : scores) {
        response->add_ids(request->candidates(s.index_).id());
        response->add_scores(s.score_);
    }
}

string RankerAppsService::requestPrefix(const FeatureRankingRequest *request) {
    stringstream ss;
    ss << "request(" << request->reqid() << "): ";
    return ss.str();
}

Rect RankerAppsService::getHotspot(const FeatureRankingRequest *request,
                                   const Mat &image) {
    if (request->interestedareas_size() > 0) {
        const Cutboard &cb = request->interestedareas(0);
        if (cb.width() != 0 && cb.height() != 0) {
            return Rect(cb.x(), cb.y(), cb.width(), cb.height());
        }
    }
    return Rect(0, 0, image.cols, image.rows);
}

int RankerAppsService::getLimit(const FeatureRankingRequest *request) {
    int limit = request->limit();
    if (limit <= 0 || limit >= request->candidates_size()) {
        limit = request->candidates_size();
    }
    return limit;
}

}
