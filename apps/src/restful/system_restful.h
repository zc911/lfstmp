/*============================================================================
 * File Name   : system_restful.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_RESTFUL_SYSTEM_H_
#define MATRIX_APPS_RESTFUL_SYSTEM_H_

#include "restful.h"
#include "services/system_service.h"

namespace dg
{

class RestSystemServiceImpl final : public RestfulService
{
public:
    RestSystemServiceImpl(const Config *config) : service_(config) {}
    virtual ~RestSystemServiceImpl() {}

    template <class socket_type>
    void Bind(SimpleWeb::Server<socket_type>& server)
    {
        bind(server, "^/ping$", "GET", service_.Ping);
        bind(server, "^/info$", "GET", service_.SystemStatus);
        bind(server, "^/instances$", "GET", service_.GetInstances);
        bind(server, "^/config$", "POST", service_.ConfigEngine);
    }

private:
    SystemAppsService service_;
};
}

#endif //MATRIX_APPS_RESTFUL_SYSTEM_H_
