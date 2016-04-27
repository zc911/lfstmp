/*============================================================================
 * File Name   : system_grpc.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_GRPC_RANKER_H_
#define MATRIX_APPS_GRPC_RANKER_H_

#include <grpc++/grpc++.h>
#include "services/system_service.h"

namespace dg 
{

class GrpcSystemServiceImpl final : public SimilarityService::Service
{
public:
    GrpcSystemServiceImpl(Config *config) : service_(config) {}
    virtual ~GrpcSystemServiceImpl() {}

private:
    RankAppsService service_;

    virtual grpc::Status Ping(grpc::ServerContext* context, const PingRequest *request, PingResponse *response) override
    {
        return service_.Ping(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    }

    virtual grpc::Status SystemStatus(grpc::ServerContext* context, const SystemStatusRequest *request, SystemStatusResponse *response) override
    {
        return service_.SystemStatus(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    }

    virtual grpc::Status GetInstances(grpc::ServerContext* context, const GetInstancesRequest *request, InstanceConfigureResponse *response) override
    {
        return service_.GetInstances(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    }

    virtual grpc::Status ConfigEngine(grpc::ServerContext* context, const InstanceConfigureRequest *request, InstanceConfigureResponse *response) override
    {
        return service_.ConfigEngine(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    }
};

}

 #endif //MATRIX_APPS_GRPC_RANKER_H_
