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

namespace dg {

class RestWitnessServiceImpl final: public RestfulService {
public:
    RestWitnessServiceImpl(const Config *config)
        : RestfulService(), service_(config) {
    }
    virtual ~RestWitnessServiceImpl() { }

    virtual void Bind(HttpServer &server) override {

        BindFunction<WitnessRequest, WitnessResponse> recBinder =
            std::bind(&WitnessAppsService::Recognize, &service_, std::placeholders::_1, std::placeholders::_2);
        BindFunction<WitnessBatchRequest, WitnessBatchResponse> batchRecBinder =
            std::bind(&WitnessAppsService::BatchRecognize, &service_, std::placeholders::_1, std::placeholders::_2);

        bind(server, "^/rec/image$", "POST", recBinder);
        bind(server, "^/rec/image/batch$", "POST", batchRecBinder);
    }

private:
    WitnessAppsService service_;
};
}

#endif //MATRIX_APPS_RESTFUL_WITNESS_H_
