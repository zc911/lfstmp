/*============================================================================
 * File Name   : ranker_grpc.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_GRPC_RANKER_H_
#define MATRIX_APPS_GRPC_RANKER_H_

#include <grpc++/grpc++.h>
#include "../model/common.pb.h"
#include "services/ranker_service.h"
#include "services/system_service.h"

using namespace ::dg::model;
namespace dg {

class GrpcRankerServiceImpl final: public SimilarityService::Service {
public:
    GrpcRankerServiceImpl(const Config *config) : service_ranker_(config),service_system_(config) { }
    virtual ~GrpcRankerServiceImpl() { }

private:
    RankerAppsService service_ranker_;
    SystemAppsService service_system_;
    virtual grpc::Status GetRankedVector(grpc::ServerContext *context,
                                         const FeatureRankingRequest *request,
                                         FeatureRankingResponse *response) override {
        MatrixError err = service_ranker_.GetRankedVector(request, response);
        return err.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;
    }
    virtual grpc::Status Ping(grpc::ServerContext* context, const PingRequest *request, PingResponse *response)
    {
        MatrixError err = service_system_.Ping(request, response);
        return err.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;
    }

    virtual grpc::Status SystemStatus(grpc::ServerContext* context, const SystemStatusRequest *request, SystemStatusResponse *response)
    {
        MatrixError err = service_system_.SystemStatus(request, response);
        return err.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;
    }

    virtual grpc::Status GetInstances(grpc::ServerContext* context, const GetInstancesRequest *request, InstanceConfigureResponse *response)
    {
        MatrixError err = service_system_.GetInstances(request, response);
        return err.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;
    }

};

}

#endif //MATRIX_APPS_GRPC_RANKER_H_
