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

class BasicGrpcService {

public:

    BasicGrpcService(Config config,
                     string addr) : config_(config),
        addr_(addr) {

    }
    virtual ~BasicGrpcService() {

    }

    virtual ::grpc::Service *service() = 0;

    void Run() {
        grpc::ServerBuilder builder;
        builder.SetMaxMessageSize(1024 * 1024 * 1024);

        builder.AddListeningPort(addr_, grpc::InsecureServerCredentials());
        builder.RegisterService(service());
        unique_ptr<grpc::Server> server(builder.BuildAndStart());

        cout << " Server(GRPC) listening on " << (int) config_.Value("System/Port")
             << endl;
        server->Wait();
    }
protected:
    Config config_;
    string addr_;
};
}

#endif //PROJECT_BASIC_GRPC_H
