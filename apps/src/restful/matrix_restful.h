/*============================================================================
 * File Name   : matrix_restful.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_RESTFUL_MATRIX_H_
#define MATRIX_APPS_RESTFUL_MATRIX_H_

#include "restful.h"
#include "services/matrix_service.h"

namespace dg
{

class RestMatrixServiceImpl final : public RestfulService
{
public:
    RestMatrixServiceImpl(const Config *config) : service_(config) {}
    virtual ~RestMatrixServiceImpl() {}

    virtual void Bind(HttpServer& server) override
    {
        BindFunction<PingRequest, PingResponse> pingBinder = std::bind(&SystemAppsService::Ping, &service_, std::placeholders::_1, std::placeholders::_2);
        BindFunction<SystemStatusRequest, SystemStatusResponse> statusBinder = std::bind(&SystemAppsService::SystemStatus, &service_, std::placeholders::_1, std::placeholders::_2);
        BindFunction<InstanceConfigureRequest, InstanceConfigureResponse> getInstBinder = std::bind(&SystemAppsService::GetInstances, &service_, std::placeholders::_1, std::placeholders::_2);
        BindFunction<GetInstancesRequest, InstanceConfigureResponse> configBinder = std::bind(&SystemAppsService::ConfigEngine, &service_, std::placeholders::_1, std::placeholders::_2);
        BindFunction<WitnessRequest, WitnessResponse> recBinder = std::bind(&WitnessAppsService::Recognize, &service_, std::placeholders::_1, std::placeholders::_2);
        BindFunction<WitnessBatchRequest, WitnessBatchResponse> batchRecBinder = std::bind(&WitnessAppsService::BatchRecognize, &service_, std::placeholders::_1, std::placeholders::_2);
        BindFunction<SkynetRequest, SkynetResponse> recVideoBinder = std::bind(&SkynetAppsService::VideoRecognize, &service_, std::placeholders::_1, std::placeholders::_2);
        BindFunction<FeatureRankingRequest, FeatureRankingResponse> rankBinder = std::bind(&RankerAppsService::GetRankedVector, &service_, std::placeholders::_1, std::placeholders::_2);

        bind(server, "^/ping$", "GET", pingBinder);
        bind(server, "^/info$", "GET", statusBinder);
        bind(server, "^/instances$", "GET", getInstBinder);
        bind(server, "^/config$", "POST", configBinder);
        bind(server, "^/rec/image$", "POST", recBinder);
        bind(server, "^/rec/image/batch$", "POST", batchRecBinder);
        bind(server, "^/rec/video$", "POST", recVideoBinder);
        bind(server, "^/rank$", "POST", rankBinder);
    }

private:
    MatrixAppsService service_;

    // virtual grpc::Status Ping(grpc::ServerContext* context, const PingRequest *request, PingResponse *response) override
    // {
    //     return service_.Ping(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    // }

    // virtual grpc::Status SystemStatus(grpc::ServerContext* context, const SystemStatusRequest *request, SystemStatusResponse *response) override
    // {
    //     return service_.SystemStatus(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    // }

    // virtual grpc::Status GetInstances(grpc::ServerContext* context, const GetInstancesRequest *request, InstanceConfigureResponse *response) override
    // {
    //     return service_.GetInstances(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    // }

    // virtual grpc::Status ConfigEngine(grpc::ServerContext* context, const InstanceConfigureRequest *request, InstanceConfigureResponse *response) override
    // {
    //     return service_.ConfigEngine(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    // }

    // virtual grpc::Status Recognize(grpc::ServerContext* context, const WitnessRequest* request, WitnessResponse* response) override
    // {
    //     return service_.Recognize(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    // }

    // virtual grpc::Status BatchRecognize(grpc::ServerContext* context, const WitnessBatchRequest* request, WitnessBatchResponse* response) override
    // {
    //     return service_.BatchRecognize(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    // }

    // virtual grpc::Status VideoRecognize(grpc::ServerContext* context, const SkynetRequest* request, SkynetResponse* response) override
    // {
    //     return service_.VideoRecognize(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    // }

    // virtual grpc::Status GetRankedVector(grpc::ServerContext* context, const FeatureRankingRequest* request, FeatureRankingResponse* response) override
    // {
    //     return service_.GetRankedVector(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
    // }
};

}

 #endif //MATRIX_APPS_RESTFUL_MATRIX_H_
