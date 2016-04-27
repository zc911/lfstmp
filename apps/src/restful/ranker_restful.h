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

namespace dg
{

class RestRankerServiceImpl final : public RestfulService
{
public:
    RestRankerServiceImpl(Config *config) : service_(config) {}
    virtual ~RestRankerServiceImpl() {}

    template <class socket_type>
    virtual void Bind(SimpleWeb::ServerBase<socket_type>& server) override
    {
        bind(server, "^/rank$", "POST", service_.GetRankedVector);
    }

private:
    RankerAppsService service_;
};
}

#endif //MATRIX_APPS_RESTFUL_RANKER_H_