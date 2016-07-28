#if true

#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "vehicle_processor_head.h"
#include "processor/vehicle_color_processor.h"
#include "file_reader.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static VehicleProcessorHead *head;
static VehicleColorProcessor *vcprocessor;
static FileReader fileReader("data/mapping/vehicle_colorhaoquan.txt");

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

TEST(VehicleColorProcessorTest, vehicleColorTest) {
    init();
    fbhelper->setBasePath("data/testimg/vehicleColor/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();
    head->process(fb);

    EXPECT_TRUE(fileReader.is_open());
    fileReader.read("=");

    FileReader result("data/testimg/vehicleColor/result.txt");
    EXPECT_TRUE(result.is_open());
    result.read(",");

    for (int i = 0; i < fb->batch_size(); ++i) {
        stringstream s;
        s << i;
        vector<string> expectColor = result.getValue(s.str());
        if (expectColor.empty()) {

            continue;
        }

        Object *obj = fb->frames()[i]->objects()[0];
        Vehicle *v = (Vehicle *)obj;
        s.str("");
        s << v->color().class_id;
        vector<string> realColor = fileReader.getValue(s.str());

        EXPECT_EQ(expectColor[0], realColor[0]);
    }

    destory();
}

#endif
