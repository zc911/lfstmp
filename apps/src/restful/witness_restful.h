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

typedef MatrixError (*BatchRecFunc)(WitnessAppsService *, const WitnessBatchRequest *, WitnessBatchResponse *);
typedef MatrixError (*RecFunc)(WitnessAppsService *, const WitnessRequest *, WitnessResponse *);

class RestWitnessServiceImpl final: public RestfulService {
public:
    RestWitnessServiceImpl(const Config *config)
        : RestfulService() {
    }
    virtual ~RestWitnessServiceImpl() { }

    void Bind(HttpServer &server, Config &config) {


        int threadNum = (int) config.Value("System/ThreadsPerGpu");

        for (int i = 0; i < threadNum; ++i) {
            cout << "init apps " << i << endl;
            WitnessAppsService *apps = new WitnessAppsService(&config, "apps_" + to_string(i));

            BindFunction<WitnessRequest, WitnessResponse> recBinder =
                std::bind(&WitnessAppsService::Recognize, apps, std::placeholders::_1, std::placeholders::_2);


            RecFunc rec_func = (RecFunc) &WitnessAppsService::Recognize;
            BatchRecFunc batch_func = (BatchRecFunc) &WitnessAppsService::BatchRecognize;

            bindFunc<WitnessAppsService, WitnessBatchRequest, WitnessBatchResponse>(server,
                                                                                    "^/rec/image/batch$",
                                                                                    "POST", batch_func);
            bindFunc(server, "^/rec/image$", "POST", rec_func);

            StartThread(apps);

        }
    }

    virtual void Bind(HttpServer &server) { };
};
}

#endif //MATRIX_APPS_RESTFUL_WITNESS_H_
