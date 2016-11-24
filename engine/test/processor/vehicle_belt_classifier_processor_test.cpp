#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "vehicle_processor_head.h"
#include "processor/vehicle_belt_classifier_processor.h"
#include "file_reader.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static VehicleProcessorHead *head;
static VehicleBeltClassifierProcessor *vbcprocessor;
static FileReader *resultReader;
static VehicleWindowProcessor *window;
static void initConfig() {
    VehicleBeltConfig mConfig;
    string baseModelPath;
#ifdef UNENCRYPTMODEL
    mConfig.is_model_encrypt = false;
    baseModelPath = "data/0/";
#else
    mConfig.is_model_encrypt = true;
    baseModelPath = "data/1/";
#endif

    mConfig.deploy_file = baseModelPath + "602.txt";
    mConfig.model_file = baseModelPath + "602.dat";

    vbcprocessor = new VehicleBeltClassifierProcessor(mConfig, true);
}

static Operation getOperation() {
    Operation op;
    op.Set( OPERATION_VEHICLE |
            OPERATION_VEHICLE_MARKER |
            OPERATION_VEHICLE_DETECT );
    return op;
}

static void init() {
    resultReader = NULL;
    head = new VehicleProcessorHead();

    fbhelper = new FrameBatchHelper(1);

    window = new VehicleWindowProcessor();
    initConfig();

    head->setNextProcessor(window->getProcessor());
    window->setNextProcessor(vbcprocessor);

}

static void destory() {
    if (head) {
        delete head;
        head = NULL;
    }
    if (resultReader) {
        delete resultReader;
        resultReader = NULL;
    }
    if (fbhelper) {
        delete fbhelper;
        fbhelper = NULL;
    }
}

TEST(VehicleBeltClassifierTest, vehicleBeltClassifierTest) {
    init();
    fbhelper->setBasePath("data/testimg/beltClassifier/");
    fbhelper->readImage(getOperation());
    head->process(fbhelper->getFrameBatch());
    FrameBatch *fb = fbhelper->getFrameBatch();
    resultReader = new FileReader("data/testimg/beltClassifier/result.txt");

    EXPECT_TRUE(resultReader->is_open());
    resultReader->read(",");

    for (int i = 0; i < fb->batch_size(); ++i) {
        vector<Object *>vehicles = fb->frames()[i]->objects();

        stringstream s;
        s << i;

        for (int j = 0; j < vehicles.size(); j++) {
            Vehicle *obj = (Vehicle *)vehicles[j];
            Vehicler *v = (Vehicler *)obj->child(OBJECT_DRIVER);

            if (v == NULL)
                continue;
            if( v->vehicler_attr_value(Vehicler::NoBelt)>0.9)
            //EXPECT_EQ(resultReader->getIntValue(s.str(),0), v->vehicler_attr_[Vehicler::Belt]) << "i = " << i << endl;
            EXPECT_EQ(resultReader->getIntValue(s.str(),j), 1) << "i = " << i << endl;
        }

    }

    destory();
}

