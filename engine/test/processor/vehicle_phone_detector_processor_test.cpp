#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "vehicle_processor_head.h"
#include "processor/vehicle_phone_detector_processor.h"
#include "file_reader.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static VehicleProcessorHead *head;
static VehiclePhoneClassifierProcessor *vbcprocessor;
static FileReader *resultReader;
static VehicleWindowProcessor *window;
static void initConfig() {
    VehicleCaffeDetectorConfig mConfig;
    string baseModelPath;
#ifdef UNENCRYPTMODEL
    mConfig.is_model_encrypt = false;
    baseModelPath = "data/0/";
#else
    mConfig.is_model_encrypt = true;
    baseModelPath = "data/1/";
#endif

    mConfig.deploy_file = baseModelPath + "604.txt";
    mConfig.model_file = baseModelPath + "604.dat";

    vbcprocessor = new VehiclePhoneClassifierProcessor(mConfig);
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

TEST(VehiclePhoneClassifierTest, vehiclePhoneClassifierTest) {
    init();
    fbhelper->setBasePath("data/testimg/phoneClassifier/");
    fbhelper->readImage(getOperation());
    head->process(fbhelper->getFrameBatch());
    FrameBatch *fb = fbhelper->getFrameBatch();
    resultReader = new FileReader("data/testimg/phoneClassifier/result.txt");

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
            if( v->vehicler_attr_value(Vehicler::Phone)>0.9)
            //EXPECT_EQ(resultReader->getIntValue(s.str(),0), v->vehicler_attr_[Vehicler::Belt]) << "i = " << i << endl;
            EXPECT_EQ(resultReader->getIntValue(s.str(),j), 1) << "i = " << i << endl;

        }

    }

    destory();
}

