#include <string>
#include <memory>
#include <iostream>
#include <glog/logging.h>
#include "config/config.h"
#include "service/witness_service.h"
#include "service/witness_service_asyn.h"

using namespace std;

using namespace dg;
using namespace dg::apps;

int main(int argc, char* argv[]) {

    google::InitGoogleLogging(argv[0]);

    Config *config = Config::GetInstance();
    config->Load("config.json");

    if ((string) config->Value("InstanceType") == "witness") {
        cout << "Instance Type: " << (string) config->Value("InstanceType")
             << endl;
        bool asyn = config->Value("System/EnableAsyn");

        BasicService *service;

        if (asyn) {
            service = new WitnessServiceAsynImpl(config);
        } else {
            service = new WitnessServiceImpl(config);
        }

        service->Run();
    } else {

    }
    return 0;
}

