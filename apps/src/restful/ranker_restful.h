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

namespace dg {

class RestRankerServiceImpl final : public RestfulService
{
public:
    RestRankerServiceImpl(const Config *config)
    : RestfulService()
    , service_(config)
    {
    }
    virtual ~RestRankerServiceImpl() {}

    virtual void Bind(HttpServer& server) override
    {
        BindFunction<FeatureRankingRequest, FeatureRankingResponse> rankBinder = std::bind(&RankerAppsService::GetRankedVector, &service_, std::placeholders::_1, std::placeholders::_2);

        bind(server, "^/rank$", "POST", rankBinder);
    }

private:
    RankerAppsService service_;
};

}

#endif //MATRIX_APPS_RESTFUL_RANKER_H_
