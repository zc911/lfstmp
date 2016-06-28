//
// Created by chenzhen on 6/3/16.
//

#ifndef PROJECT_BASIC_GRPC_H
#define PROJECT_BASIC_GRPC_H

#include <string>
#include <grpc++/grpc++.h>
#include "../model/common.pb.h"
#include "services/witness_service.h"
#include "services/engine_pool.h"

using namespace std;
using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;

namespace dg {

template<class EngineType>
class BasicGrpcService {

public:

    BasicGrpcService(Config config,
                     string addr,
                     MatrixEnginesPool <EngineType> *engine_pool) : config_(config),
                                                                    addr_(addr),
                                                                    engine_pool_(engine_pool) {

    }

    virtual ~BasicGrpcService() {

    }

    virtual ::grpc::Service *service() = 0;

    void Run() {

        engine_pool_->Run();

        grpc::ServerBuilder builder;
        builder.AddListeningPort(addr_, grpc::InsecureServerCredentials());
        builder.RegisterService(service());
        unique_ptr<grpc::Server> server(builder.BuildAndStart());

        cout << typeid(EngineType).name() << " Server(GRPC) listening on " << (int) config_.Value("System/Port")
            << endl;
        string name = typeid(EngineType).name();
        name.find("Witness",0);

        server->Wait();
    }
    virtual void warmUp(int n){
        string imgdata=ReadStringFromFile("warmup.dat","rb");
        WitnessRequest protobufRequestMessage;
        WitnessResponse protobufResponseMessage;
        protobufRequestMessage.mutable_image()->mutable_data()->set_bindata(imgdata);
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

protected:
    Config config_;
    string addr_;
    MatrixEnginesPool <EngineType> *engine_pool_;
};
template<class MessageType>
class BasicGrpcClient {
public:
    BasicGrpcClient(Config config, MessagePool <MessageType> *message_pool)
        : config_(config), message_pool_(message_pool) { }
    virtual ~BasicGrpcClient() {

    }
    virtual void Run() {
        message_pool_->Run();
    }
protected:
    Config config_;
    MessagePool <MessageType> *message_pool_;
};
}

#endif //PROJECT_BASIC_GRPC_H
