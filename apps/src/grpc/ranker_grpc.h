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
#include "common.pb.h"
#include "services/ranker_service.h"
#include "services/system_service.h"
#include "basic_grpc.h"

using namespace ::dg::model;
namespace dg {

class GrpcRankerServiceImpl final: public BasicGrpcService, public SimilarityService::Service {
public:

    GrpcRankerServiceImpl(Config config, string addr)
        : BasicGrpcService(config, addr) {
            service_ = new RankerAppsService(&config,"WitnessAppsService"); 
 }

    virtual ~GrpcRankerServiceImpl() { 
      delete service_;
}
    virtual ::grpc::Service *service() {
        return this;
    };

    virtual grpc::Status GetRankedVector(grpc::ServerContext *context,
                                         const FeatureRankingRequest *request,
                                         FeatureRankingResponse *response) override {

        cout << "[GRPC] ========================" << endl;
        cout << "[GRPC] Get rank request, thread id: " << this_thread::get_id() << endl;
      /*  CallData data;

        data.func = [request, response, &data]() -> MatrixError {
          return (bind(&RankerAppsService::GetRankedVector,
                       (RankerAppsService *) data.apps,
                       placeholders::_1,
                       placeholders::_2))(request,
                                          response);
        };

        service_pool_->enqueue(&data);
        MatrixError error = data.Wait();
        return error.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;
*/
        MatrixError error = service_->GetRankedVector(request,response);
        return error.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;

    }
    RankerAppsService *service_;

};

}

#endif //MATRIX_APPS_GRPC_RANKER_H_
