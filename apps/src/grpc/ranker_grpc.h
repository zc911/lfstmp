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
#include "service/ranker_service.h"

namespace dg 
{

class GrpcSystemImpl final : public SystemService::Service
{
public:
    GrpcSystemImpl(Config *config) : service_(config) {}
    virtual ~GrpcSystemImpl() {}

private:
    SystemService service_;

    virtual grpc::Status GetRankedVector(grpc::ServerContext* context, const FeatureRankingRequest* request, FeatureRankingResponse* response) override
    {
        return service_.GetRankedVector(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    }
};

}

 #endif //MATRIX_APPS_GRPC_RANKER_H_