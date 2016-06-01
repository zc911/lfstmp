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

#include "Simple-Web-Server/server_http.hpp"

using namespace std;
using namespace dg;


string getServerAddress(Config *config, int userPort = 0) {
    if (userPort != 0) {
        cout << "Use command line port instead of config file value: " << endl;
        config->AddEntry("System/Port", AnyConversion(userPort));
    }

    return (string) config->Value("System/Ip") + ":"
        + (string) config->Value("System/Port");
}

void serveGrpc(Config *config, int userPort = 0) {
    string instType = (string) config->Value("InstanceType");
    cout << "Instance type: " << instType << endl;
    string address = getServerAddress(config, userPort);

    if (instType == "witness") {
        GrpcWitnessServiceAsynImpl *service = new GrpcWitnessServiceAsynImpl(config);
        cout << "Server(RRPC AYSN) listening on " << address << endl;
        service->Run();
    } else if (instType == "ranker") {
        grpc::Service *service = NULL;
        service = new GrpcRankerServiceImpl(config);
        grpc::ServerBuilder builder;
        builder.AddListeningPort(address, grpc::InsecureServerCredentials());
        builder.RegisterService(service);
        unique_ptr<grpc::Server> server(builder.BuildAndStart());
        cout << "Server(GRPC) listening on " << address << endl;
        server->Wait();
    } else {
        cout << "unknown instance type: " << instType << endl;
        return;
    }

}

void serveHttp(const Config *config, int userPort = 0) {
    string instType = (string) config->Value("InstanceType");
    cout << "Instance type: " << instType << endl;

    int port = (int) config->Value("System/Port");
    if (userPort) {
        port = userPort;
    }

//    RestfulService *service = NULL;
    if (instType == "witness") {
        RestWitnessServiceImpl *service = new RestWitnessServiceImpl(config);
        SimpleWeb::Server<SimpleWeb::HTTP> server(port, 5);  //at port with 1 thread
        service->Bind(server, const_cast<Config&>(*config));
        cout << "Server(RESTFUL) listening on " << port << endl;
        server.start();

    } else if (instType == "ranker") {
//        service = new RestRankerServiceImpl(config);
    } else {
        cout << "unknown instance type: " << instType << endl;
        return;
    }






}

int main(int argc, char *argv[]) {

    google::InitGoogleLogging(argv[0]);

    google::ParseCommandLineFlags(&argc, &argv, true);

    int userPort = 0;
    if (argc >= 2) {
        userPort = atoi(argv[1]);
    }
    string configFile = "config.json";
    if(argc >= 3){
        configFile = argv[2];
    }

    Config *config = new Config();


    config->Load(configFile);


    string protocolType = (string) config->Value("ProtocolType");
    cout << "Protocol type: " << protocolType << endl;
    if (protocolType == "rpc") {
        serveGrpc(config, userPort);
    } else if (protocolType == "restful") {
        serveHttp(config, userPort);
    } else {
        cout << "unknown protocol type: " << protocolType << endl;
    }

    return 0;
}

