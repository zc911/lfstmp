#if true

#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "vehicle_processor_head.h"
#include "processor/vehicle_classifier_processor.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static VehicleProcessorHead *head;
static VehicleClassifierProcessor *vcfprocessor;

static void initConfig() {
    VehicleCaffeClassifier::VehicleCaffeConfig config;
    config.is_model_encrypt = false;
    string basePath = "data/models/";
    for (int i = 0; i < 8; ++i) {
        char index[2] = {0};
        index[0] = '0' + i;
        config.deploy_file = basePath + "10" + string(index) + ".txt";
        config.model_file = basePath + "10" + string(index) + ".dat";
    }
    vector<VehicleCaffeClassifier::VehicleCaffeConfig> configs;
    configs.push_back(config);
    vcfprocessor = new VehicleClassifierProcessor(configs);
}

static void init() {
    initConfig();
    head = new VehicleProcessorHead();
    fbhelper = new FrameBatchHelper(1);
    head->setNextProcessor(vcfprocessor);
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
            OPERATION_VEHICLE_STYLE |
            OPERATION_VEHICLE_DETECT );
    return op;
}

TEST(VehicleClassifierProcessorTest, VehicleClassifierTest) {
    init();
    fbhelper->setBasePath("data/testimg/vehicleClassifier/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();
    head->process(fb);

    int expectId[] = {
            387, 15, -1, 446, 507, 32, 15, 15, 2645, 847
    };

    for (int i = 0; i < fb->batch_size(); ++i) {
        Object *obj = fb->frames()[i]->objects()[0];
        Vehicle *v = (Vehicle *)obj;
        EXPECT_EQ(expectId[i], v->class_id());
    }

    delete fbhelper;
    fbhelper = new FrameBatchHelper(1);
    fbhelper->setBasePath("data/testimg/vehicleClassifier/");
    fbhelper->readImage(getOperation());

    for (int i = 0; i < fbhelper->getFrameBatch()->frames().size(); ++i) {
        Frame *f = fbhelper->getFrameBatch()->frames()[i];
        head->getProcessor()->Update(f);
        EXPECT_EQ(0, f->objects().size());
    }

    destory();
}

#endif
