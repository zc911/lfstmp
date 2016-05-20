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

using namespace ::dg::model;
namespace dg {

class GrpcWitnessServiceImpl final: public WitnessService::Service {
public:
    GrpcWitnessServiceImpl(const Config *config) : service_(config) { }
    virtual ~GrpcWitnessServiceImpl() { }

private:
    WitnessAppsService service_;


    virtual grpc::Status Recognize(grpc::ServerContext *context,
                                   const WitnessRequest *request,
                                   WitnessResponse *response) override {
        MatrixError err = service_.Recognize(request, response);
        return err.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;
    }

    virtual grpc::Status BatchRecognize(grpc::ServerContext *context,
                                        const WitnessBatchRequest *request,
                                        WitnessBatchResponse *response) override {
        std::thread::id threadId = std::this_thread::get_id();
        cout << "Batch rec in thread id: "<< hex << threadId << endl;
        MatrixError err = service_.BatchRecognize(request, response);
        return err.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;
    }
};

}

#endif //MATRIX_APPS_GRPC_WITNESS_H_
