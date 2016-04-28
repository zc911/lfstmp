/*============================================================================
 * File Name   : matrix_service.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_MATRIX_SERVICE_H_
#define MATRIX_APPS_MATRIX_SERVICE_H_

#include "config.h"
#include "model/matrix.grpc.pb.h"

namespace dg
{

class MatrixAppsService
{
public:
    MatrixAppsService(const Config *config);
    virtual ~MatrixAppsService();

    bool Ping(const PingRequest *request, PingResponse *response);

    bool SystemStatus(const SystemStatusRequest *request, SystemStatusResponse *response);

    bool GetInstances(const GetInstancesRequest *request, InstanceConfigureResponse *response);

    bool ConfigEngine(const InstanceConfigureRequest *request, InstanceConfigureResponse *response);

    bool Recognize(const WitnessRequest *request, WitnessResponse *response);

    bool BatchRecognize(const WitnessBatchRequest *request, WitnessBatchResponse *response);

    bool VideoRecognize(const SkynetRequest *request, SkynetResponse *response);

    bool GetRankedVector(const FeatureRankingRequest* request, FeatureRankingResponse* response);

private:
    const Config *config_;

};

}

#endif //MATRIX_APPS_MATRIX_SERVICE_H_
