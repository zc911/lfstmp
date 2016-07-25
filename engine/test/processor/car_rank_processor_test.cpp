#if false

#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "vehicle_processor_head.h"
#include "processor/vehicle_color_processor.h"
#include "processor/car_rank_processor.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static VehicleProcessorHead *head;
static VehicleColorProcessor *vcprocessor;
static CarRankProcessor *crprocessor;

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
    crprocessor = new CarRankProcessor();
    head->setNextProcessor(crprocessor);
}

static void destory() {
    /**
    if (head) {
        delete head;
        head = NULL;
    }
     **/

    if (fbhelper) {
        delete fbhelper;
        fbhelper = NULL;
    }
}

static Operation getOperation() {
    Operation op;
    op.Set( OPERATION_VEHICLE |
            OPERATION_VEHICLE_DETECT );
    return op;
}

TEST(CarRankProcessorTest, carRankTest) {

    init();
    EXPECT_EQ(1, -1);

}

#endif
