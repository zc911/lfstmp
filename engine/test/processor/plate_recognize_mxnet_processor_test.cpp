#if true

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
            OPERATION_VEHICLE_PLATE |
            OPERATION_VEHICLE_DETECT );
    return op;
}

static string expectPlate[] = {
        "蒙GUT288",
        "京P0SG96",
        "",
        "贵A01P89",
        "贵JAB353",
        "苏DW7782",
        "",
        "京PM8P28",
        "京NFA261",
        "京N9D3H9"
};

TEST(PlateRecognizeMxnetTest, plateRecognizeTest) {
    init();
    fbhelper->setBasePath("data/testimg/plateRecognize/recognize/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();

    head->process(fb);

    for (int i = 0; i < fb->batch_size(); ++i) {
        Object *obj = fb->frames()[i]->objects()[0];
        Vehicle *v = (Vehicle *)obj;
        if (v->plates().empty()) {
            EXPECT_TRUE(expectPlate[i] == "");
        }
        else {
            EXPECT_TRUE(expectPlate[i] == v->plates()[0].plate_num);
        }
    }

    delete fbhelper;
    fbhelper = new FrameBatchHelper(1);
    fbhelper->setBasePath("data/testimg/markerClassifier/");
    fbhelper->readImage(getOperation());

    for (int i = 0; i < fb->batch_size(); ++i) {
        head->getProcessor()->Update(fb->frames()[i]);
        EXPECT_EQ(0, fb->frames()[i]->objects().size()) << "i = " << i << endl;
    }

    destory();
}

TEST(PlateRecognizeMxnetTest, handleWithNoDectorTest) {
    initConfig();
    fbhelper = new FrameBatchHelper(1);
    fbhelper->setBasePath("data/testimg/plateRecognize/recognize/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();

    for (int i = 0; i < fb->batch_size(); ++i) {
        Frame *frame = fb->frames()[i];
        Vehicle *vehicle = new Vehicle(OBJECT_CAR);
        Mat tmp = frame->payload()->data();
        Detection d;
        d.box = Rect(0, 0, tmp.cols, tmp.rows);
        vehicle->set_id(i);
        vehicle->set_image(tmp);
        Object *obj = static_cast<Object *>(vehicle);
        frame->put_object(obj);
    }

    prmprocessor->Update(fb);

    for (int i = 0; i < fb->batch_size(); ++i) {
        prmprocessor->Update(fb->frames()[i]);
    }

    for (int i = 0; i < fb->batch_size(); ++i) {
        Object *obj = fb->frames()[i]->objects()[0];
        Vehicle *v = (Vehicle *)obj;
        if (v->plates().empty()) {
            EXPECT_TRUE(expectPlate[i] == "");
        }
        else {
            EXPECT_TRUE(expectPlate[i] == v->plates()[0].plate_num);
        }
    }

    delete prmprocessor;
}

TEST(PlateRecognizeMxnetTest, plateColorTest) {
    initConfig();
    fbhelper = new FrameBatchHelper(1);
    fbhelper->setBasePath("data/testimg/plateRecognize/color/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();

    for (int i = 0; i < fb->batch_size(); ++i) {
        Frame *frame = fb->frames()[i];
        Vehicle *vehicle = new Vehicle(OBJECT_CAR);
        Mat tmp = frame->payload()->data();
        Detection d;
        d.box = Rect(0, 0, tmp.cols, tmp.rows);
        vehicle->set_id(i);
        vehicle->set_image(tmp);
        Object *obj = static_cast<Object *>(vehicle);
        frame->put_object(obj);
    }

    prmprocessor->Update(fb);

    int expectPlateColor[] = {
            2, 6, 0, 0, 0
    };

    for (int i = 0; i < fb->batch_size(); ++i) {
        Vehicle *obj = (Vehicle*)fb->frames()[i]->objects()[0];
        EXPECT_EQ(expectPlateColor[i], obj->plates()[0].color_id);
    }

    delete prmprocessor;
}

/**
TEST(PlateRecognizeMxnetTest, plateSizeTest) {
    initConfig();
    fbhelper = new FrameBatchHelper(1);
    fbhelper->setBasePath("data/testimg/plateRecognize/number/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();

    for (int i = 0; i < fb->batch_size(); ++i) {
        Frame *frame = fb->frames()[i];
        Vehicle *vehicle = new Vehicle(OBJECT_CAR);
        Mat tmp = frame->payload()->data();
        Detection d;
        d.box = Rect(0, 0, tmp.cols, tmp.rows);
        vehicle->set_id(i);
        vehicle->set_image(tmp);
        Object *obj = static_cast<Object *>(vehicle);
        frame->put_object(obj);
    }

    prmprocessor->Update(fb);

    int expectPlateNumber[] = {
           1, 5, 2, 7, 11, 8
    };

    fbhelper->printFrame();

    for (int i = 0; i < fb->batch_size(); ++i) {
        Vehicle *obj = (Vehicle*)fb->frames()[i]->objects()[0];
        EXPECT_EQ(expectPlateNumber[i], obj->plates().size());
    }

    delete prmprocessor;
}
 */

#endif
