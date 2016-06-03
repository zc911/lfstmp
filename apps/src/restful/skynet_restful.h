/*============================================================================
 * File Name   : skynet_restful.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_RESTFUL_SKYNET_H_
#define MATRIX_APPS_RESTFUL_SKYNET_H_

#include "restful.h"
#include "services/skynet_service.h"

namespace dg {

//class RestSkynetServiceImpl final : public RestfulService<>
//{
//public:
//    RestSkynetServiceImpl(const Config *config) : service_(config) {}
//    virtual ~RestSkynetServiceImpl() {}
//
//    virtual void Bind(HttpServer& server) override
//    {
//        BindFunction<SkynetRequest, SkynetResponse> recVideoBinder = std::bind(&SkynetAppsService::VideoRecognize, &service_, std::placeholders::_1, std::placeholders::_2);
//
//        bind(server, "^/rec/video$", "POST", recVideoBinder);
//    }
//
//private:
//    SkynetAppsService service_;
//};
}

#endif //MATRIX_APPS_RESTFUL_SKYNET_H_
