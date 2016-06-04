/*============================================================================
 * File Name   : system_restful.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/
/*
#ifndef MATRIX_APPS_RESTFUL_SYSTEM_H_
#define MATRIX_APPS_RESTFUL_SYSTEM_H_

#include "restful.h"
#include "services/system_service.h"

namespace dg {

class RestSystemServiceImpl final : public RestfulService
{
public:
    RestSystemServiceImpl(const Config *config)
    : RestfulService()
    , service_(config)
    {
    }
    virtual ~RestSystemServiceImpl() {}

    virtual void Bind(HttpServer& server) override
    {
        BindFunction<PingRequest, PingResponse> pingBinder = std::bind(&SystemAppsService::Ping, &service_, std::placeholders::_1, std::placeholders::_2);
        BindFunction<SystemStatusRequest, SystemStatusResponse> statusBinder = std::bind(&SystemAppsService::SystemStatus, &service_, std::placeholders::_1, std::placeholders::_2);
        BindFunction<InstanceConfigureRequest, InstanceConfigureResponse> getInstBinder = std::bind(&SystemAppsService::GetInstances, &service_, std::placeholders::_1, std::placeholders::_2);
        BindFunction<GetInstancesRequest, InstanceConfigureResponse> configBinder = std::bind(&SystemAppsService::ConfigEngine, &service_, std::placeholders::_1, std::placeholders::_2);

        bind(server, "^/ping$", "GET", pingBinder);
        bind(server, "^/info$", "GET", statusBinder);
        bind(server, "^/instances$", "GET", getInstBinder);
        bind(server, "^/config$", "POST", configBinder);
    }

private:
    SystemAppsService service_;
};
}

#endif //MATRIX_APPS_RESTFUL_SYSTEM_H_
*/
