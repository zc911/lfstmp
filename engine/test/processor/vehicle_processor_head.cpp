#include "vehicle_processor_head.h"

using namespace dg;

VehicleProcessorHead::VehicleProcessorHead() {
    init();
}

VehicleProcessorHead::~VehicleProcessorHead() {
    if (processor) {
        delete processor;
        processor = NULL;
    }
}

void VehicleProcessorHead::init() {
    processor = new VehicleMultiTypeDetectorProcessor(getConfig());
}

VehicleCaffeDetectorConfig VehicleProcessorHead::getConfig() {
    VehicleCaffeDetectorConfig config;
    string basePath;

#ifdef UNENCRYPTMODEL
    config.is_model_encrypt = false;
    basePath = "data/0/";
#else
    config.is_model_encrypt = true;
    basePath = "data/1/";
#endif

    config.target_max_size = 600;
    config.target_min_size = 400;
    config.deploy_file = config.confirm_deploy_file =
            basePath + "300.txt";
    config.model_file = config.confirm_deploy_file =
            basePath + "300.dat";
    return config;
}
