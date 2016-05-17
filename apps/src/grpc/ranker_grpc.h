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

using namespace ::dg::model;
namespace dg {

class GrpcRankerServiceImpl final: public SimilarityService::Service {
public:
    GrpcRankerServiceImpl(const Config *config) : service_(config) { }
    virtual ~GrpcRankerServiceImpl() { }

private:
    RankerAppsService service_;

    virtual grpc::Status GetRankedVector(grpc::ServerContext *context,
                                         const FeatureRankingRequest *request,
                                         FeatureRankingResponse *response) override {
        MatrixError err = service_.GetRankedVector(request, response);
        return err.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;
    }
};

}

#endif //MATRIX_APPS_GRPC_RANKER_H_
