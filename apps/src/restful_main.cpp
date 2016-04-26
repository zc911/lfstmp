#include <string>
#include <memory>
#include <iostream>
#include <glog/logging.h>
#include "config/config.h"
#include "grpc/witness_service.h"
#include "grpc/witness_service_asyn.h"

using namespace std;

using namespace dg;

int main(int argc, char* argv[]) {

    google::InitGoogleLogging(argv[0]);

    Config *config = Config::GetInstance();
    config->Load("config.json");

    return 0;
}

