#if false

#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "vehicle_processor_head.h"
#include "processor/plate_recognize_mxnet_processor.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static VehicleProcessorHead *head;
static PlateRecognizeMxnetProcessor *prmprocessor;

static void initConfig() {
    LPDRConfig_S config;
    string basePath = "data/models/";
    config.fcnnSymbolFile = basePath + "801.txt";
    config.fcnnParamFile = basePath + "801.dat";
    config.pregSymbolFile = basePath + "802.txt";
    config.pregParamFile = basePath + "802.dat";
    config.chrecogSymbolFile = basePath + "800.txt";
    config.chrecogParamFile = basePath + "800.dat";
    config.roipSymbolFile = basePath + "803.txt";
    config.roipParamFile = basePath + "803.dat";
    config.rpnSymbolFile = basePath + "804.txt";
    config.rpnParamFile = basePath + "804.dat";

    config.is_model_encrypt = false;
    config.dwDevID = 0;
    config.dwDevType = 2;
    config.imageSH = 800;
    config.imageSW = 800;
    config.numsPlates = 2;
    config.numsProposal = 20;
    config.plateSH = 400;
    config.plateSW = 300;

    prmprocessor = new PlateRecognizeMxnetProcessor(&config);
}

static void init() {
    initConfig();
    head = new VehicleProcessorHead();
    fbhelper = new FrameBatchHelper(1);
    head->setNextProcessor(prmprocessor);
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
            OPERATION_VEHICLE_PLATE |
            OPERATION_VEHICLE_DETECT );
    return op;
}

TEST(PlateRecognizeMxnetTest, PlateRecognizeTest) {
    cout << "YUYYYYYYYYYYYYYYYYYYYYYYYY" << endl;
    init();
    fbhelper->setBasePath("data/testimg/test/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();

    head->process(fb);
//    prmprocessor->Update(fb);

    fbhelper->printFrame();

    /**
    int expectId[] = {
            2207, 506, 206
    };

    for (int i = 0; i < fb->batch_size(); ++i) {
        Object *obj = fb->frames()[i]->objects()[0];
        Vehicle *v = (Vehicle *)obj;
        EXPECT_EQ(expectId[i], v->class_id());
    }
     **/

//    destory();
}

#endif
