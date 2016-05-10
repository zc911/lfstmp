/*============================================================================
 * File Name   : matrix_grpc.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_GRPC_MATRIX_H_
#define MATRIX_APPS_GRPC_MATRIX_H_

#include <grpc++/grpc++.h>
#include "services/matrix_service.h"

namespace dg {

class GrpcMatrixServiceImpl final : public MatrixService::Service
{
public:
    GrpcMatrixServiceImpl(const Config *config) : service_(config) {}
    virtual ~GrpcMatrixServiceImpl() {}

private:
    MatrixAppsService service_;

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

    virtual grpc::Status Recognize(grpc::ServerContext* context, const WitnessRequest* request, WitnessResponse* response) override
    {
        return service_.Recognize(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    }

    virtual grpc::Status BatchRecognize(grpc::ServerContext* context, const WitnessBatchRequest* request, WitnessBatchResponse* response) override
    {
        return service_.BatchRecognize(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    }

    virtual grpc::Status VideoRecognize(grpc::ServerContext* context, const SkynetRequest* request, SkynetResponse* response) override
    {
        return service_.VideoRecognize(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    }

    virtual grpc::Status GetRankedVector(grpc::ServerContext* context, const FeatureRankingRequest* request, FeatureRankingResponse* response) override
    {
        return service_.GetRankedVector(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    }
};

}

#endif //MATRIX_APPS_GRPC_MATRIX_H_
