//
// Created by chenzhen on 6/3/16.
//

#ifndef PROJECT_BASIC_GRPC_H
#define PROJECT_BASIC_GRPC_H

#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <cstring>
#include <string>
#include <grpc++/grpc++.h>
#include "common.pb.h"
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
        try {
            int serv_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            struct sockaddr_in serv_addr;
            memset(&serv_addr, 0, sizeof(serv_addr));
            serv_addr.sin_family = AF_INET;
            serv_addr.sin_addr.s_addr = inet_addr("0.0.0.0");
            serv_addr.sin_port = htons((int) config_.Value("System/Port"));
            int isOccupied = bind(serv_sock, (struct sockaddr *) &serv_addr, sizeof(serv_addr));
            close(serv_sock);
            if (isOccupied) {
                cout << " Listening on port " << (int) config_.Value("System/Port") << " failed" << endl;
                throw std::runtime_error("bind: Address already in use");
            }
        }
        catch (...) {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_exceptions.push_back(std::current_exception());
            return;
        }
        grpc::ServerBuilder builder;
        builder.SetMaxMessageSize(1024 * 1024 * 1024);

        builder.AddListeningPort(addr_, grpc::InsecureServerCredentials());
        builder.RegisterService(service());
        unique_ptr<grpc::Server> server(builder.BuildAndStart());

        cout << " Server(GRPC) listening on " << (int) config_.Value("System/Port")
             << endl;
        server->Wait();
    }

    std::vector<std::exception_ptr> getExceptions() const {
        return g_exceptions;
    }
protected:
    Config config_;
    string addr_;
    std::mutex g_mutex;
    std::vector<std::exception_ptr>  g_exceptions;
};
}

#endif //PROJECT_BASIC_GRPC_H
