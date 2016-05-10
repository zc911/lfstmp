#include <string>
#include <memory>
#include <iostream>
#include <glog/logging.h>
#include <grpc++/grpc++.h>

#define BOOST_SPIRIT_THREADSAFE
#include "config.h"

#include "grpc/witness_grpc.h"
#include "grpc/ranker_grpc.h"

#include "restful/witness_restful.h"
#include "restful/ranker_restful.h"

#include "server_http.hpp"

using namespace std;
using namespace dg;

void serveGrpc(const Config* config) {
    string instType = (string) config->Value("InstanceType");
    cout << "Instance type: " << instType << endl;

    grpc::Service *service = NULL;
    if (instType == "witness") {
        service = new GrpcWitnessServiceImpl(config);
    } else if (instType == "ranker") {
        service = new GrpcRankerServiceImpl(config);
    } else {
        cout << "unknown instance type: " << instType << endl;
        return;
    }

    string address = (string) config->Value("System/Ip") + ":"
            + (string) config->Value("System/Port");

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(service);
    unique_ptr<grpc::Server> server(builder.BuildAndStart());
    cout << "Server listening on " << address << endl;
    server->Wait();
}

void serveHttp(const Config* config) {
    string instType = (string) config->Value("InstanceType");
    cout << "Instance type: " << instType << endl;

    RestfulService *service = NULL;
    if (instType == "witness") {
        service = new RestWitnessServiceImpl(config);
    } else if (instType == "ranker") {
        service = new RestRankerServiceImpl(config);
    } else {
        cout << "unknown instance type: " << instType << endl;
        return;
    }

    int port = (int) config->Value("System/Port");
    SimpleWeb::Server<SimpleWeb::HTTP> server(port, 1);  //at port with 1 thread
    service->Bind(server);

    cout << "Server listening on " << port << endl;
    server.start();
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    Config *config = new Config();
    config->Load("config.json");

    string protocolType = (string) config->Value("ProtocolType");
    cout << "Protocol type: " << protocolType << endl;
    if (protocolType == "rpc") {
        serveGrpc(config);
    } else if (protocolType == "restful") {
        serveHttp(config);
    } else {
        cout << "unknown protocol type: " << protocolType << endl;
    }

    return 0;
}

