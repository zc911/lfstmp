/*============================================================================
 * File Name   : ranker_service.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_RANKER_SERVICE_H_
#define MATRIX_APPS_RANKER_SERVICE_H_

#include <opencv2/core/core.hpp>
#include "config.h"
#include "config/config_val.h"
#include "../model/common.pb.h"
#include "../model/ranker.grpc.pb.h" //from apps
#include "engine/rank_engine.h"
using namespace cv;
using namespace std;

namespace dg {
using namespace ::dg::model;

class RankerAppsService {
public:
    RankerAppsService(const Config *config, string name);
    virtual ~RankerAppsService();

    MatrixError GetRankedVector(const FeatureRankingRequest *request,
                                FeatureRankingResponse *response);

private:
    string name_;
    const Config *config_;
    CarRankEngine car_ranker_;
    FaceRankEngine face_ranker_;
    MatrixError (RankerAppsService::*getRankedDefaultVector)(const FeatureRankingRequest *,
                                   FeatureRankingResponse *);
    MatrixError getRankedCarVector(const FeatureRankingRequest *request,
                                   FeatureRankingResponse *response);

    MatrixError getRankedFaceVector(const FeatureRankingRequest *request,
                                    FeatureRankingResponse *response);
    MatrixError getRankedAllVector(const FeatureRankingRequest *request,
                                   FeatureRankingResponse *response);
    MatrixError getCarScoredVector(vector<Score> &scores, const FeatureRankingRequest *request,
                                   FeatureRankingResponse *response);
    MatrixError getFaceScoredVector(vector<Score> &scores, const FeatureRankingRequest *request,
                                    FeatureRankingResponse *response);
    static void sortAndFillResponse(const FeatureRankingRequest *request,
                                    vector<Score> &scores,
                                    FeatureRankingResponse *response);

    static string requestPrefix(const FeatureRankingRequest *request);

    static Rect getHotspot(const FeatureRankingRequest *request,
                           const Mat &image);

    static int getLimit(const FeatureRankingRequest *request);

    template<typename F>
    static MatrixError extractFeatures(const FeatureRankingRequest *request,
                                       vector<F> &features, int limits) {
        MatrixError err;
        if (request->candidates_size() <= 0) {
            err.set_code(-1);
            err.set_message("no candidates in request context");
            return err;
        }
        if (request->candidates_size() > limits) {
            err.set_code(-1);
            err.set_message("too many candidates");
            return err;
        }
        for (int i = 0; i < request->candidates_size(); i++) {
            string featureStr = request->candidates(i).feature();
            if (featureStr.size() <= 0) {
                err.set_code(-1);
                err.set_message("invalid candidate");
                return err;
            }

            F feature;
            if (!feature.Deserialize(featureStr)) {
                LOG(ERROR) << "Invalid feature string, id: " << request->candidates(i).id() << endl;
                err.set_code(-1);
                err.set_message("invalid candidate feature");
                return err;
            }

            features.push_back(feature);
        }

        return err;
    }
};

}

#endif //MATRIX_APPS_RANKER_SERVICE_H_
