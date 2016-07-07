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
        warmUp(config_.Value("System/ThreadsPerGpu"));
        grpc::ServerBuilder builder;
        builder.SetMaxMessageSize(1024 * 1024 * 1024);

        builder.AddListeningPort(addr_, grpc::InsecureServerCredentials());
        builder.RegisterService(service());
        unique_ptr<grpc::Server> server(builder.BuildAndStart());

        cout << typeid(EngineType).name() << " Server(GRPC) listening on " << (int) config_.Value("System/Port")
            << endl;
        bool EnabledDetection = (bool) config_.Value("Feature/Vehicle/EnableDetection");
        if (!EnabledDetection)
            warmUp(config_.Value("System/ThreadsPerGpu"));
        server->Wait();
    }

    virtual void warmUp(int n) {

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
