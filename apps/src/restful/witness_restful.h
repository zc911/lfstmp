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
        : RestfulService() {
//        , service_1_(config, "restService1"), service_2_(config, "restService1")
    }
    virtual ~RestWitnessServiceImpl() { }

    void Bind(HttpServer &server, Config &config) {


        int threadNum = (int) config.Value("System/ThreadsPerGpu");

        for (int i = 0; i < threadNum; ++i) {
            cout << "init apps " << i << endl;
            WitnessAppsService *apps = new WitnessAppsService(&config, "apps_" + to_string(i));

            BindFunction<WitnessRequest, WitnessResponse> recBinder =
                std::bind(&WitnessAppsService::Recognize, apps, std::placeholders::_1, std::placeholders::_2);

//            BindFunction<WitnessBatchRequest, WitnessBatchResponse> batchRecBinder =
//                std::bind(&WitnessAppsService::BatchRecognize, apps, std::placeholders::_1, std::placeholders::_2);

//            bind(server, "^/rec/image$", "POST", recBinder);
//            std::function<MatrixError(const WitnessBatchRequest *, WitnessBatchResponse *)>
//                f = &WitnessAppsService::BatchRecognize;

            typedef MatrixError (*FUNC)(WitnessAppsService *, const WitnessBatchRequest *, WitnessBatchResponse *);
            FUNC func = (FUNC) &WitnessAppsService::BatchRecognize;

            bindFunc<WitnessAppsService, WitnessBatchRequest, WitnessBatchResponse>(server,
                                                                "^/rec/image/batch$",
                                                                "POST", func);

            StartThread(apps);

        }
//        InitServer(server, config);
//        BindFunction<WitnessRequest, WitnessResponse> recBinder =
//            std::bind(&WitnessAppsService::Recognize, &service_1_, std::placeholders::_1, std::placeholders::_2);
//        BindFunction<WitnessBatchRequest, WitnessBatchResponse> batchRecBinder =
//            std::bind(&WitnessAppsService::BatchRecognize, &service_1_, std::placeholders::_1, std::placeholders::_2);

//        bind(server, "^/rec/image$", "POST", recBinder);
//        bind(server, "^/rec/image/batch$", "POST", batchRecBinder);
//    }
    }

    virtual void Bind(HttpServer &server) { };
//private:
//    WitnessAppsService service_1_;
//
//    WitnessAppsService service_2_;
};
}

#endif //MATRIX_APPS_RESTFUL_WITNESS_H_
