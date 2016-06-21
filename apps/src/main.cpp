#include <string>
#include <memory>
#include <iostream>
#include <glog/logging.h>
#include <grpc++/grpc++.h>

#define BOOST_SPIRIT_THREADSAFE
#include <gflags/gflags.h>
#include <curl/curl.h>

#include "config.h"

#include "grpc/witness_grpc.h"
#include "grpc/ranker_grpc.h"

#include "restful/witness_restful.h"
#include "restful/ranker_restful.h"
#include "services/engine_pool.h"
#include "watchdog/watch_dog.h"
#include "grpc/system_grpc.h"
#include "services/witness_bucket.h"

using namespace std;
using namespace dg;
WitnessBucket WitnessBucket::instance_;

string getServerAddress(Config *config, int userPort = 0) {
    if (userPort != 0) {
        cout << "Use command line port instead of config file value" << endl;
        config->AddEntry("System/Port", AnyConversion(userPort));
    }

    return (string) config->Value("System/Ip") + ":" + (string) config->Value("System/Port");
}


void serveWitness(Config *config, int userPort = 0) {
    string protocolType = (string) config->Value("ProtocolType");
    cout << "Protocol type: " << protocolType << endl;
    string address = getServerAddress(config, userPort);

    MessagePool<StorageRequest> *msg_pool = new MessagePool<StorageRequest>(config);
    WitnessBucket::Instance().SetMaxSize(100);
    msg_pool->Run();
    SpringGrpcClientImpl *client = new SpringGrpcClientImpl(*config, msg_pool);
    std::thread test(&SpringGrpcClientImpl::Run, client);
    MatrixEnginesPool<WitnessAppsService> *engine_pool = new MatrixEnginesPool<WitnessAppsService>(config);
    engine_pool->Run();
    std::thread network_th_(networkInfo, &rx, &tx);

    if (protocolType == "restful") {
        RestWitnessServiceImpl *service = new RestWitnessServiceImpl(*config, address, engine_pool);
        service->Run();
    } else if (protocolType == "rpc") {
        GrpcWitnessServiceImpl *service = new GrpcWitnessServiceImpl(*config, address, engine_pool);
        std::thread t1(&GrpcWitnessServiceImpl::Run, service);
        string address2 = getServerAddress(config, (int) config->Value("System/Port") + 1);
        MatrixEnginesPool<SystemAppsService> *engine_pool1 = new MatrixEnginesPool<SystemAppsService>(config);
        engine_pool1->Run();
        GrpcSystemServiceImpl *system_service = new GrpcSystemServiceImpl(*config, address2, engine_pool1);
        std::thread t2(&GrpcSystemServiceImpl::Run, system_service);
        t1.join();
        t2.join();
    } else if (protocolType == "restful|rpc" || protocolType == "rpc|restful") {
        GrpcWitnessServiceImpl *service = new GrpcWitnessServiceImpl(*config, address, engine_pool);
        std::thread t1(&GrpcWitnessServiceImpl::Run, service);
        string address2 = getServerAddress(config, (int) config->Value("System/Port") + 1);
        RestWitnessServiceImpl *service2 = new RestWitnessServiceImpl(*config, address2, engine_pool);
        std::thread t2(&RestWitnessServiceImpl::Run, service2);
        t1.join();
        t2.join();
    } else {
        cout << "Invalid protocol, should be rpc, restful or rpc|restful" << endl;
        exit(-1);
    }
    test.join();
    network_th_.join();

}

void serveRanker(Config *config, int userPort = 0) {
    string protocolType = (string) config->Value("ProtocolType");
    cout << "Protocol type: " << protocolType << endl;
    string address = getServerAddress(config, userPort);

    MatrixEnginesPool<RankerAppsService> *engine_pool = new MatrixEnginesPool<RankerAppsService>(config);
    engine_pool->Run();
    if (protocolType == "restful") {
        RestRankerServiceImpl *service = new RestRankerServiceImpl(*config, address, engine_pool);
        service->Run();
    } else if (protocolType == "rpc") {
        GrpcRankerServiceImpl *service = new GrpcRankerServiceImpl(*config, address, engine_pool);
        std::thread t1(&GrpcRankerServiceImpl::Run, service);
        string address2 = getServerAddress(config, (int) config->Value("System/Port") + 1);
        MatrixEnginesPool<SystemAppsService> *engine_pool1 = new MatrixEnginesPool<SystemAppsService>(config);
        engine_pool1->Run();
        GrpcSystemServiceImpl *system_service = new GrpcSystemServiceImpl(*config, address2, engine_pool1);
        std::thread t2(&GrpcSystemServiceImpl::Run, system_service);
        t1.join();
        t2.join();
    } else if (protocolType == "restful|rpc" || protocolType == "rpc|restful") {
        GrpcRankerServiceImpl *service = new GrpcRankerServiceImpl(*config, address, engine_pool);
        std::thread t1(&GrpcRankerServiceImpl::Run, service);
        string address2 = getServerAddress(config, (int) config->Value("System/Port") + 1);
        RestRankerServiceImpl *service2 = new RestRankerServiceImpl(*config, address2, engine_pool);
        std::thread t2(&RestRankerServiceImpl::Run, service2);
        t1.join();
        t2.join();
    } else {
        cout << "Invalid protocol, should be rpc, restful or rpc|restful" << endl;
        exit(-1);
    }
}


DEFINE_int32(port, 0, "Service port number, will overwite the value defined in config file");
DEFINE_string(config, "config.json", "Config file path");

int main(int argc, char *argv[]) {

    google::InitGoogleLogging(argv[0]);
//    StartDogMonitor();
//  if (CheckHardware()) {
 //      return -1;
 // }

    google::SetUsageMessage("Usage: " + string(argv[0]) + " [--port=6500] [--config=config.json]");
    google::SetVersionString("0.2.4");
    google::ParseCommandLineFlags(&argc, &argv, false);

    // init curl in the main thread
    // see https://curl.haxx.se/libcurl/c/curl_easy_init.html
    curl_global_init(CURL_GLOBAL_ALL);

    Config *config = new Config();
    config->Load(FLAGS_config);

    string instType = (string) config->Value("InstanceType");

    if (instType == "witness") {
        serveWitness(config, FLAGS_port);
    } else if (instType == "ranker") {
        serveRanker(config, FLAGS_port);
    } else {
        cout << "Invalid instance type , should be either witness or ranker." << endl;
        return -1;
    }

    return 0;
}

