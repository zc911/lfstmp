/*============================================================================
 * File Name   : skynet_grpc.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_GRPC_SKYNET_H_
#define MATRIX_APPS_GRPC_SKYNET_H_

#include <grpc++/grpc++.h>
#include "service/skynet_service.h"

namespace dg 
{

class GrpcSkynetImpl final : public SkynetService::Service
{
public:
    GrpcSkynetImpl(Config *config) : service_(config) {}
    virtual ~GrpcSkynetImpl() {}

private:
    SkynetService service_;

    virtual grpc::Status VideoRecognize(grpc::ServerContext* context, const SkynetRequest* request, SkynetResponse* response) override
    {
        return service_.VideoRecognize(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    }
};

}

 #endif //MATRIX_APPS_GRPC_SKYNET_H_