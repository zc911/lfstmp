/*============================================================================
 * File Name   : witness_restful.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_RESTFUL_WITNESS_H_
#define MATRIX_APPS_RESTFUL_WITNESS_H_

#include "restful.h"
#include "services/witness_service.h"

namespace dg
{

class RestWitnessServiceImpl final : public RestfulService
{
public:
    RestWitnessServiceImpl(Config *config) : service_(config) {}
    virtual ~RestWitnessServiceImpl() {}

    template <class socket_type>
    virtual void Bind(SimpleWeb::ServerBase<socket_type>& server) override
    {
        bind(server, "^/rec/image$", "POST", service_.Recognize);
        bind(server, "^/rec/image/batch$", "POST", service_.BatchRecognize);
    }

private:
    WitnessAppsService service_;
};
}

#endif //MATRIX_APPS_RESTFUL_WITNESS_H_