#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "vehicle_processor_head.h"
#include "processor/vehicle_color_processor.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static VehicleProcessorHead *head;
static VehicleColorProcessor *vcprocessor;

static void initConfig() {
    CaffeVehicleColorClassifier::VehicleColorConfig config;
    config.is_model_encrypt = false;
    config.deploy_file = "data/models/200.txt";
    config.model_file = "data/models/200.dat";
    vector<CaffeVehicleColorClassifier::VehicleColorConfig> configs;
    configs.push_back(config);
    vcprocessor = new VehicleColorProcessor(configs);
}

static void init() {
    initConfig();
    head = new VehicleProcessorHead();
    fbhelper = new FrameBatchHelper(1);
    head->setNextProcessor(vcprocessor);
}

static void destory() {
   if (head) {
       delete head;
       head = NULL;
   }

   if (fbhelper) {
       delete fbhelper;
       fbhelper = NULL;
   }
}

static Operation getOperation() {
    Operation op;
    op.Set( OPERATION_VEHICLE |
            OPERATION_VEHICLE_COLOR |
            OPERATION_VEHICLE_DETECT );
    return op;
}

TEST(VehicleColorProcessorTest, VehicleColorTest) {
    init();
    fbhelper->setBasePath("data/testimg/test/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();
    head->process(fb);

    vcprocessor->Update(fb);

    fbhelper->printFrame();

//    destory();
}
