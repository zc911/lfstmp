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

namespace dg
{

class RestSkynetServiceImpl final : public RestfulService
{
public:
    RestSkynetServiceImpl(Config *config) : service_(config) {}
    virtual ~RestSkynetServiceImpl() {}

    template <class socket_type>
    virtual void Bind(SimpleWeb::ServerBase<socket_type>& server) override
    {
        bind(server, "^/rec/video$", "POST", service_.VideoRecognize);
    }

private:
    SkynetAppsService service_;
};
}

#endif //MATRIX_APPS_RESTFUL_SKYNET_H_