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
        service_ = new RankerAppsService(&config, "RankerAppsService");
    }

    virtual ~GrpcRankerServiceImpl() {
        delete service_;
    }
    virtual ::grpc::Service *service() {
        return this;
    };

    virtual grpc::Status RankImage(grpc::ServerContext *context,
                                   const RankImageRequest *request,
                                   RankImageResponse *response) {
        cout << "This service is not implemented now, use RankFeature instead." << endl;
        return grpc::Status::CANCELLED;
    }

    virtual grpc::Status RankFeature(grpc::ServerContext *context,
                                     const RankFeatureRequest *request,
                                     RankFeatureResponse *response) {

        cout << "[GRPC] ========================" << endl;
        cout << "[GRPC] Get rank request, thread id: " << this_thread::get_id() << endl;
        MatrixError error = service_->RankFeature(request, response);
        response->mutable_context()->set_message(error.message());
        response->mutable_context()->set_status(std::to_string(error.code()));
        return error.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;
    }
    virtual ::grpc::Status AddFeatures(::grpc::ServerContext *context,
                                       const ::dg::model::AddFeaturesRequest *request,
                                       ::dg::model::AddFeaturesResponse *response) {
        cout << "[GRPC] ========================" << endl;
        cout << "[GRPC] Add features in runtime" << endl;
        MatrixError error = service_->AddFeatures(request, response);
        response->mutable_context()->set_message(error.message());
        response->mutable_context()->set_status(std::to_string(error.code()));
        return error.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;

    }

    virtual ::grpc::Status GetImageContent(::grpc::ServerContext *context,
                                           const ::dg::model::GetImageContentRequest *request,
                                           ::dg::model::GetImageContentResponse *response) {
        MatrixError error = service_->GetImageContent(request, response);
        return error.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;

    }

    virtual ::grpc::Status Search(::grpc::ServerContext *context,
                                  const ::dg::model::SearchRequest *request,
                                  ::dg::model::SearchResponse *response) {
        MatrixError err = service_->Search(request, response);
        return err.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;
    }

    virtual ::grpc::Status RankRepoSize(::grpc::ServerContext *context,
                                        const ::dg::model::RankRepoSizeRequest *request,
                                        ::dg::model::RankRepoSizeResponse *response) {

        MatrixError err = service_->RepoSize(request, response);
        return err.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;

    }


    /**
     * @Deprecated
     */
    virtual grpc::Status GetRankedVector(grpc::ServerContext *context,
                                         const FeatureRankingRequest *request,
                                         FeatureRankingResponse *response) override {
        cout << "This service is Deprecated, use RankImage or RankFeature instead. " << endl;
        MatrixError error;
        error.set_code(-1);
        error.set_message("This service is Deprecated, use RankImage or RankFeature instead. ");
        return grpc::Status::CANCELLED;

    }
    RankerAppsService *service_;

};

}

#endif //MATRIX_APPS_GRPC_RANKER_H_