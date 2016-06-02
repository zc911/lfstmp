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

typedef MatrixError (*RecFunc)(WitnessAppsService *, const WitnessRequest *, WitnessResponse *);
typedef MatrixError (*BatchRecFunc)(WitnessAppsService *, const WitnessBatchRequest *, WitnessBatchResponse *);

class RestWitnessServiceImpl final: public RestfulService {
public:
    RestWitnessServiceImpl(Config config,
                           string addr,
                           MatrixEnginesPool<WitnessAppsService> *engine_pool)
        : RestfulService(engine_pool), config_(config) {

    }

    virtual ~RestWitnessServiceImpl() { }

    void Bind(HttpServer &server) {

        RecFunc rec_func = (RecFunc) &WitnessAppsService::Recognize;
        bindFunc<WitnessAppsService, WitnessRequest, WitnessResponse>(server, "^/rec/image$",
                                                                      "POST", rec_func);
        BatchRecFunc batch_func = (BatchRecFunc) &WitnessAppsService::BatchRecognize;
        bindFunc<WitnessAppsService, WitnessBatchRequest, WitnessBatchResponse>(server,
                                                                                "/rec/image/batch$",
                                                                                "POST",
                                                                                batch_func);

    }

    void Run() {
        int port = (int) config_.Value("System/Port");
        int gpuNum = (int) config_.Value("System/GpuNum");
        gpuNum = gpuNum == 0 ? 1 : gpuNum;

        int threadsPerGpu = (int) config_.Value("System/ThreadsPerGpu");
        threadsPerGpu = threadsPerGpu == 0 ? 1 : threadsPerGpu;

        int threadNum = gpuNum * threadsPerGpu;

        SimpleWeb::Server<SimpleWeb::HTTP> server(port, threadNum);  //at port with 1 thread
        Bind(server);
        if(engine_pool_ == NULL){
            LOG(ERROR) << "Engine pool not initialized" << endl;
        }
        engine_pool_->Run();
        cout << "Server(RESTFUL) listening on " << port << endl;
        server.start();
    }

private:
    Config config_;
};
}

#endif //MATRIX_APPS_RESTFUL_WITNESS_H_
