#include "vehicle_processor_head.h"
#include "algorithm_factory.h"

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
    dgvehicle::AlgorithmFactory::GetInstance()->Initialize("data/dgvehicle", 0, false);
    processor = new VehicleMultiTypeDetectorProcessor(false, false);
    dgvehicle::AlgorithmFactory::GetInstance()->ReleaseUselessModel();
}

VehicleWindowProcessor::VehicleWindowProcessor() {
    init();
}

VehicleWindowProcessor::~VehicleWindowProcessor() {
    if (processor) {
        delete processor;
        processor = NULL;
    }
}
void VehicleWindowProcessor::init() {
 //   processor = new VehicleWindowDetectorProcessor(getConfig());
    dgvehicle::AlgorithmFactory::GetInstance()->Initialize("data/dgvehicle", 0, false);
    processor = new VehicleWindowDetectorProcessor();
    dgvehicle::AlgorithmFactory::GetInstance()->ReleaseUselessModel();
}

//VehicleCaffeDetectorConfig VehicleWindowProcessor::getConfig() {
//    VehicleCaffeDetectorConfig config;
//    string basePath;
//
//#ifdef UNENCRYPTMODEL
//    config.is_model_encrypt = false;
//    basePath = "data/0/";
//#else
//    config.is_model_encrypt = true;
//    basePath = "data/1/";
//#endif
//
//    config.target_max_size = 160;
//    config.target_min_size = 80;
//    config.deploy_file = config.confirm_deploy_file =
//                             basePath + "701.txt";
//    config.model_file = config.confirm_deploy_file =
//                            basePath + "701.dat";
//    return config;
//}
