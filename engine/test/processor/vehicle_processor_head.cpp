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
    config.is_model_encrypt = false;
    config.deploy_file = config.confirm_deploy_file =
            "data/models/300.txt";
    config.model_file = config.confirm_deploy_file =
            "data/models/300.dat";
    return config;
}
