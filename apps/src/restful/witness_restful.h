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
    RestWitnessServiceImpl(const Config *config)
               : RestfulService()
               , service_(config) 
               , rec_binder_(std::bind(&WitnessAppsService::Recognize, &service_, std::placeholders::_1, std::placeholders::_2))
               , batch_rec_binder_(std::bind(&WitnessAppsService::BatchRecognize, &service_, std::placeholders::_1, std::placeholders::_2))
    {
    }
    virtual ~RestWitnessServiceImpl() {}

    virtual void Bind(HttpServer& server) override
    {
        rec_binder_.Bind(server, "^/rec/image$", "POST");
        batch_rec_binder_.Bind(server, "^/rec/image/batch$", "POST");
    }

private:
    WitnessAppsService service_;
    RestfulBinder<WitnessRequest, WitnessResponse> rec_binder_;
    RestfulBinder<WitnessBatchRequest, WitnessBatchResponse> batch_rec_binder_;
};
}

#endif //MATRIX_APPS_RESTFUL_WITNESS_H_
