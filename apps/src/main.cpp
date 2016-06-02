#include <string>
#include <memory>
#include <iostream>
#include <glog/logging.h>
#include <grpc++/grpc++.h>

#define BOOST_SPIRIT_THREADSAFE
#include <curl/curl.h>
#include "config.h"

#include "grpc/witness_grpc.h"
#include "grpc/ranker_grpc.h"

#include "restful/witness_restful.h"
#include "restful/ranker_restful.h"
#include "services/engine_pool.h"

#include "Simple-Web-Server/server_http.hpp"

using namespace std;
using namespace dg;


string getServerAddress(Config *config, int userPort = 0) {
    if (userPort != 0) {
        cout << "Use command line port instead of config file value: " << endl;
        config->AddEntry("System/Port", AnyConversion(userPort));
    }

    return (string) config->Value("System/Ip") + ":" + (string) config->Value("System/Port");
}

void serveGrpcWitness(GrpcWitnessServiceImpl *service){
    service->Run();
}


void serveWitness(Config *config, int userPort = 0) {
    string protocolType = (string) config->Value("ProtocolType");
    cout << "Protocol type: " << protocolType << endl;
    string address = getServerAddress(config, userPort);


    MatrixEnginesPool<WitnessAppsService> *engine_pool = new MatrixEnginesPool<WitnessAppsService>(config);
    engine_pool->Run();

    if (protocolType == "restful") {
        RestWitnessServiceImpl *service = new RestWitnessServiceImpl(*config, address, engine_pool);
        service->Run();
    } else if (protocolType == "rpc") {
        GrpcWitnessServiceImpl *service = new GrpcWitnessServiceImpl(*config, address, engine_pool);
        service->Run();
    } else if(protocolType == "restful|rpc" || protocolType == "rpc|restful"){
        GrpcWitnessServiceImpl *service = new GrpcWitnessServiceImpl(*config, address, engine_pool);
        std::thread t1(&GrpcWitnessServiceImpl::Run, service);
        string address2 = getServerAddress(config, (int)config->Value("System/Port") + 1);
        RestWitnessServiceImpl *service2 = new RestWitnessServiceImpl(*config, address2, engine_pool);
        std::thread t2(&RestWitnessServiceImpl::Run, service2);
        t1.join();
        t2.join();
    }

}

int main(int argc, char *argv[]) {

    google::InitGoogleLogging(argv[0]);

    google::ParseCommandLineFlags(&argc, &argv, true);
    // init curl in the main thread
    // see https://curl.haxx.se/libcurl/c/curl_easy_init.html
    curl_global_init(CURL_GLOBAL_ALL);

    int userPort = 0;
    if (argc >= 2) {
        userPort = atoi(argv[1]);
    }
    string configFile = "config.json";
    if (argc >= 3) {
        configFile = argv[2];
    }

    Config *config = new Config();
    config->Load(configFile);

    string instType = (string) config->Value("InstanceType");

    if (instType == "witness") {
        serveWitness(config, userPort);
    } else if (instType == "ranker") {

    } else {
        cout << "Instance type invalid, should be either witness or ranker." << endl;
        return -1;
    }


//    string protocolType = (string) config->Value("ProtocolType");
//    cout << "Protocol type: " << protocolType << endl;
//    if (protocolType == "rpc") {
//        serveGrpc(config, userPort);
//    } else if (protocolType == "restful") {
//        serveHttp(config, userPort);
//    } else {
//        cout << "unknown protocol type: " << protocolType << endl;
//    }

    return 0;
}

