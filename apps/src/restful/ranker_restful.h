/*============================================================================
 * File Name   : ranker_restful.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description :
 * ==========================================================================*/

#ifndef MATRIX_APPS_RESTFUL_RANKER_H_
#define MATRIX_APPS_RESTFUL_RANKER_H_
#include "restful.h"
#include "services/ranker_service.h"
#include "services/system_service.h"

namespace dg {


typedef MatrixError (*RankFunc)(RankerAppsService *, const FeatureRankingRequest *, FeatureRankingResponse *);

class RestRankerServiceImpl final: public RestfulService {

 public:

    RestRankerServiceImpl(Config config,
                          string addr)
        : RestfulService(config) {
        service_ = new RankerAppsService(&config, "RankerAppsService");
    }

    virtual ~RestRankerServiceImpl() { delete service_; }

    void Bind(HttpServer &server) {

        std::function<MatrixError(const RankFeatureRequest *, RankFeatureResponse *)> rankBinder =
            std::bind(&RankerAppsService::RankFeature, service_, std::placeholders::_1, std::placeholders::_2);

        std::function<MatrixError(const AddFeaturesRequest *, AddFeaturesResponse *)> addFeaturesBinder =
            std::bind(&RankerAppsService::AddFeatures, service_, std::placeholders::_1, std::placeholders::_2);

        std::function<MatrixError(const GetImageContentRequest *, GetImageContentResponse *)> getImageContentBinder =
            std::bind(&RankerAppsService::GetImageContent, service_, std::placeholders::_1, std::placeholders::_2);

        bindFunc<RankFeatureRequest, RankFeatureResponse>(server,
                                                          "/rank$",
                                                          "POST",
                                                          rankBinder);

        bindFunc<AddFeaturesRequest, AddFeaturesResponse>(server,
                                                          "/rank/add",
                                                          "POST",
                                                          addFeaturesBinder);

        bindFunc<GetImageContentRequest, GetImageContentResponse>(server,
                                                                  "/rank/getImageContent",
                                                                  "POST",
                                                                  getImageContentBinder);

    }


 private:
    RankerAppsService *service_;

};

}

#endif //MATRIX_APPS_RESTFUL_RANKER_H_