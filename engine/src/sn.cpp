//
// Created by chenzhen on 6/30/16.
//
#include <string>
#include "io/ringbuffer.h"
#include "config.h"
#include "engine/skynet_engine.h"
using namespace std;
using namespace dg;


int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, false);

    Config *config = new Config();
    config->Load("config.json");

    SkynetEngine *engine = new SkynetEngine(*config);
    engine->Run();

}