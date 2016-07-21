//
// Created by chenzhen on 5/25/16.
//

#include "witness_grpc.h"
#include "debug_util.h"

namespace dg {

GrpcWitnessServiceImpl::GrpcWitnessServiceImpl(Config config,
                                               string addr,
                           ServicePool<WitnessAppsService,WitnessEngine> *service_pool)
    : BasicGrpcService(config, addr, service_pool) {

    RepoService::GetInstance()->Init(config);

}
GrpcWitnessServiceImpl::~GrpcWitnessServiceImpl() {
}

grpc::Status GrpcWitnessServiceImpl::Recognize(grpc::ServerContext *context,
                                               const WitnessRequest *request,
                                               WitnessResponse *response) {


    VLOG(VLOG_SERVICE) << "[GRPC] ========================" << endl;
    VLOG(VLOG_SERVICE) << "[GRPC] Get rec request, session id: " << request->context().sessionid() << endl;

    struct timeval start, finish;
    gettimeofday(&start, NULL);

    CallData data;
    data.func = [request, response, &data]() -> MatrixError {
      return (bind(&WitnessAppsService::Recognize,
                   (WitnessAppsService *) data.apps,
                   placeholders::_1,
                   placeholders::_2))(request,
                                      response);
    };
    service_pool_->enqueue(&data);
    MatrixError error = data.Wait();

    gettimeofday(&finish, NULL);
    VLOG(VLOG_SERVICE)
    << "[GRPC] Rec session id " << request->context().sessionid() << " and total cost: " << TimeCostInMs(start, finish)
        << endl;
    VLOG(VLOG_SERVICE) << "[GRPC] ========================" << endl;

    return error.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;

}
grpc::Status GrpcWitnessServiceImpl::Index(grpc::ServerContext *context,
                                           const IndexRequest *request,
                                           IndexResponse *response) {

    MatrixError error = RepoService::GetInstance()->Index(request, response);
    if (error.code() != 0) {
        return grpc::Status::CANCELLED;
    }
    return grpc::Status::OK;


}

grpc::Status GrpcWitnessServiceImpl::IndexTxt(grpc::ServerContext *context,
                                              const IndexTxtRequest *request,
                                              IndexTxtResponse *response) {

    MatrixError error = RepoService::GetInstance()->IndexTxt(request, response);
    if (error.code() != 0) {
        return grpc::Status::CANCELLED;
    }
    return grpc::Status::OK;

}
grpc::Status GrpcWitnessServiceImpl::BatchRecognize(grpc::ServerContext *context,
                                                    const WitnessBatchRequest *request,
                                                    WitnessBatchResponse *response) {

    VLOG(VLOG_SERVICE) << "[GRPC] ========================" << endl;
    VLOG(VLOG_SERVICE) << "[GRPC] Get batch rec request, session id: " << request->context().sessionid() << endl;
    struct timeval start, finish;
    gettimeofday(&start, NULL);

    CallData data;
    data.func = [request, response, &data]() -> MatrixError {
      return (bind(&WitnessAppsService::BatchRecognize,
                   (WitnessAppsService *) data.apps,
                   placeholders::_1,
                   placeholders::_2))(request,
                                      response);
    };

    service_pool_->enqueue(&data);
    MatrixError error = data.Wait();

    gettimeofday(&finish, NULL);
    VLOG(VLOG_SERVICE) << "[GRPC] Batch rec session id " << request->context().sessionid() << " and total cost: "
        << TimeCostInMs(start, finish)
        << endl;

    VLOG(VLOG_SERVICE) << "[GRPC] ========================" << endl;
    return error.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;

}

}

