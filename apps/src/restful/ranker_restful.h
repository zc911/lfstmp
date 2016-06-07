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

class RestRankerServiceImpl final : public RestfulService
{
public:
    RestRankerServiceImpl(const Config *config)
    : RestfulService()
    , service_ranker_(config),service_system_(config)
    {
    }
    virtual ~RestRankerServiceImpl() {}

    virtual void Bind(HttpServer& server) override
    {
        BindFunction<FeatureRankingRequest, FeatureRankingResponse> rankBinder = std::bind(&RankerAppsService::GetRankedVector, &service_ranker_, std::placeholders::_1, std::placeholders::_2);
        BindFunction<PingRequest, PingResponse> pingBinder = std::bind(&SystemAppsService::Ping, &service_system_, std::placeholders::_1, std::placeholders::_2);
        BindFunction<SystemStatusRequest, SystemStatusResponse> statusBinder = std::bind(&SystemAppsService::SystemStatus, &service_system_, std::placeholders::_1, std::placeholders::_2);
   //     BindFunction<InstanceConfigureRequest, InstanceConfigureResponse> getInstBinder = std::bind(&SystemAppsService::GetInstances, &service_system_, std::placeholders::_1, std::placeholders::_2);

        bind(server, "^/rank$", "POST", rankBinder);
        bind(server, "^/ping$", "GET", pingBinder);
        bind(server, "^/info$", "GET", statusBinder);
  //      bind(server, "^/instances$", "GET", getInstBinder);
    }

private:
    RankerAppsService service_ranker_;
    SystemAppsService service_system_;

};

}

#endif //MATRIX_APPS_RESTFUL_RANKER_H_
