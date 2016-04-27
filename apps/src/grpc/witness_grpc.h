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

#include <grpc++/grpc++.h>
#include "services/witness_service.h"

namespace dg 
{

class GrpcWitnessServiceImpl final : public WitnessService::Service 
{
public:
    GrpcWitnessServiceImpl(Config *config) : service_(config) {}
    virtual ~GrpcWitnessServiceImpl() {}

private:
    WitnessAppsService service_;

    virtual grpc::Status Recognize(grpc::ServerContext* context, const WitnessRequest* request, WitnessResponse* response) override
    {
        return service_.Recognize(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    }

    virtual grpc::Status BatchRecognize(grpc::ServerContext* context, const WitnessBatchRequest* request, WitnessBatchResponse* response) override
    {
        return service_.BatchRecognize(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    }
};

}

#endif //MATRIX_APPS_GRPC_WITNESS_H_