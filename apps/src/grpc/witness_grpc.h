/*============================================================================
 * File Name   : witness_grpc.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_GRPC_WITNESS_H_
#define MATRIX_APPS_GRPC_WITNESS_H_

#include <thread>
#include <mutex>
#include <grpc++/grpc++.h>
#include "../model/common.pb.h"
#include "services/witness_service.h"
#include "services/system_service.h"
#include "services/engine_pool.h"
#include "basic_grpc.h"

using namespace ::dg::model;
using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;

namespace dg {


class GrpcWitnessServiceImpl final: public BasicGrpcService, public WitnessService::Service {
public:
    GrpcWitnessServiceImpl(Config config, string addr);
    virtual ~GrpcWitnessServiceImpl();
    virtual ::grpc::Service *service() {
        return this;
    };
private:
    virtual grpc::Status Recognize(grpc::ServerContext *context,
                                   const WitnessRequest *request,
                                   WitnessResponse *response);
    virtual grpc::Status Index(grpc::ServerContext *context,
                               const IndexRequest *request,
                               IndexResponse *response);
    virtual grpc::Status IndexTxt(grpc::ServerContext *context,
                                  const IndexTxtRequest *request,
                                  IndexTxtResponse *response);

    virtual grpc::Status BatchRecognize(grpc::ServerContext *context,
                                        const WitnessBatchRequest *request,
                                        WitnessBatchResponse *response);
    //virtual grpc::Status Ping(grpc::ServerContext *context,const PingRequest *request,PingResponse *response);
    //  virtual grpc::Status SystemStatus(grpc::ServerContext *context,const SystemStatusRequest *request,SystemStatusResponse *response);
    WitnessAppsService *service_;

};

}

#endif //MATRIX_APPS_GRPC_WITNESS_H_
