/*============================================================================
 * File Name   : witness_grpc.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_GRPC_WITNESS_H_
#define MATRIX_APPS_GRPC_WITNESS_H_

#include <thread>
#include <mutex>
#include <grpc++/grpc++.h>
#include "../model/common.pb.h"
#include "services/witness_service.h"
#include "services/system_service.h"
#include "services/engine_pool.h"
#include "basic_grpc.h"

using namespace ::dg::model;
using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;

namespace dg {


class GrpcWitnessServiceImpl final: public BasicGrpcService<WitnessAppsService>, public WitnessService::Service {
public:
    GrpcWitnessServiceImpl(Config config, string addr, MatrixEnginesPool<WitnessAppsService> *engine_pool);
    virtual ~GrpcWitnessServiceImpl();
    virtual ::grpc::Service *service() {
        return this;
    };
private:
    virtual grpc::Status Recognize(grpc::ServerContext *context,
                                   const WitnessRequest *request,
                                   WitnessResponse *response);
    virtual grpc::Status Index(grpc::ServerContext *context,
                                   const IndexRequest *request,
                                   IndexResponse *response);
    virtual grpc::Status IndexTxt(grpc::ServerContext *context,
                               const IndexTxtRequest *request,
                               IndexTxtResponse *response);

    virtual grpc::Status BatchRecognize(grpc::ServerContext *context,
                                        const WitnessBatchRequest *request,
                                        WitnessBatchResponse *response);
    virtual void warmUp(int n){
        string imgdata=ReadStringFromFile("warmup.dat","rb");
        WitnessRequest protobufRequestMessage;
        WitnessResponse protobufResponseMessage;
        protobufRequestMessage.mutable_image()->mutable_data()->set_bindata(imgdata);
        protobufRequestMessage.mutable_image()->mutable_witnessmetadata()->set_sensorurl("http://127.0.0.1");
        WitnessRequestContext *ctx = protobufRequestMessage.mutable_context();
        ctx->mutable_functions()->Add(1);
        ctx->mutable_functions()->Add(2);
        ctx->mutable_functions()->Add(3);
        ctx->mutable_functions()->Add(4);
        ctx->mutable_functions()->Add(5);
        ctx->mutable_functions()->Add(6);
        ctx->mutable_functions()->Add(7);
        ctx->set_type(REC_TYPE_VEHICLE);

        for(int i=0;i<n;i++) {
            CallData data;
            typedef MatrixError (*RecFunc)(WitnessAppsService *, const WitnessRequest *, WitnessResponse *);
            RecFunc rec_func = (RecFunc) &WitnessAppsService::Recognize;
            data.func = [rec_func, &protobufRequestMessage, &protobufResponseMessage, &data]() -> MatrixError {
              return (bind(rec_func, (WitnessAppsService *) data.apps,
                           placeholders::_1,
                           placeholders::_2))(&protobufRequestMessage,
                                              &protobufResponseMessage);
            };

            if (engine_pool_ == NULL) {
                LOG(ERROR) << "Engine pool not initailized. " << endl;
                return;
            }
            engine_pool_->enqueue(&data);

            MatrixError error = data.Wait();
        }

    }
    //virtual grpc::Status Ping(grpc::ServerContext *context,const PingRequest *request,PingResponse *response);
  //  virtual grpc::Status SystemStatus(grpc::ServerContext *context,const SystemStatusRequest *request,SystemStatusResponse *response);
};

//class GrpcWitnessServiceAsynImpl final: public IGrpcWitnessService, public WitnessService::Service {
//
//public:
//    GrpcWitnessServiceAsynImpl( Config *config);
//    ~GrpcWitnessServiceAsynImpl();
//
//    // There is no shutdown handling in this code.
//    void Run();
//
//public:
//    // Class encompasing the state and logic needed to serve a request.
//    class CallData {
//    public:
//        // Take in the "service" instance (in this case representing an asynchronous
//        // server) and the completion queue "cq" used for asynchronous communication
//        // with the gRPC runtime.
//        CallData(WitnessService::AsyncService *service, WitnessAppsService *witness_apps,
//                 ServerCompletionQueue *cq, bool batchMode);
//
//        void Proceed(WitnessAppsService *witness_apps);
//
//    private:
//
//        bool batch_mode_;
//        // The means of communication with the gRPC runtime for an asynchronous
//        // server.
//        WitnessService::AsyncService *service_;
//        // The producer-consumer queue where for asynchronous server notifications.
////        WitnessAppsService *witness_apps_;
//        ServerCompletionQueue *cq_;
//        // Context for the rpc, allowing to tweak aspects of it such as the use
//        // of compression, authentication, as well as to send metadata back to the
//        // client.
//        ServerContext ctx_;
//
//        // What we get from the client.
//        WitnessBatchRequest batch_request_;
//        // What we send back to the client.
//        WitnessBatchResponse batch_reply_;
//
//        WitnessRequest request_;
//        WitnessResponse reply_;
//
//        // The means to get back to the client.
//        ServerAsyncResponseWriter<WitnessResponse> responder_;
//        ServerAsyncResponseWriter<WitnessBatchResponse> batch_responder_;
//
//
//        // Let's implement a tiny state machine with the following states.
//        enum CallStatus {
//            CREATE,
//            PROCESS,
//            FINISH
//        };
//
//        CallStatus status_;  // The current serving state.
//    };
//
//    // This can be run in multiple threads if needed.
//    void HandleRpcs(WitnessAppsService *witness_apps_);
//
////    Config *config_;
//    string addr_;
//    Config *config_;

////    WitnessAppsService *witness_apps_1_;
////    WitnessAppsService *witness_apps_2_;
////    WitnessAppsService *witness_apps_3_;
////    WitnessAppsService *witness_apps_4_;
////    WitnessAppsService *witness_apps_5_;
//    std::unique_ptr<ServerCompletionQueue> cq_;
//    WitnessService::AsyncService service_;
//    std::unique_ptr<Server> server_;
//
//    volatile int which_apps_;
//};
//
}

#endif //MATRIX_APPS_GRPC_WITNESS_H_
