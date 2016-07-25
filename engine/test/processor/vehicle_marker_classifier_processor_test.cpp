#if true

#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "vehicle_processor_head.h"
#include "processor/vehicle_marker_classifier_processor.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static VehicleProcessorHead *head;
static VehicleMarkerClassifierProcessor *vmcprocessor;

static void initConfig() {
    WindowCaffeDetector::WindowCaffeConfig wConfig;
    MarkerCaffeClassifier::MarkerConfig mConfig;

    wConfig.is_model_encrypt = false;
    wConfig.deploy_file = "data/models/700.txt";
    wConfig.model_file = "data/models/700.dat";

    mConfig.is_model_encrypt = false;
    mConfig.deploy_file = "data/models/600.txt";
    mConfig.model_file = "data/models/600.dat";

    vmcprocessor = new VehicleMarkerClassifierProcessor(wConfig, mConfig);
}

static Operation getOperation() {
    Operation op;
    op.Set( OPERATION_VEHICLE |
            OPERATION_VEHICLE_MARKER |
            OPERATION_VEHICLE_DETECT );
    return op;
}

static void init() {
    initConfig();
    head = new VehicleProcessorHead();
    fbhelper = new FrameBatchHelper(1);
    head->setNextProcessor(vmcprocessor);
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

TEST(VehicleMarkerClassifierTest, markerClassifierTest) {
    init();

    fbhelper->setBasePath("data/testimg/markerClassifier/");
    fbhelper->readImage(getOperation());
    head->process(fbhelper->getFrameBatch());
    FrameBatch *fb = fbhelper->getFrameBatch();
    int expectNumber[] = {
            3, 4, 3, 5, 1, 2, 7, 4, 3, 0, 0
    };

    for (int i = 0; i < fb->batch_size(); ++i) {
        Vehicle *obj = (Vehicle *)fb->frames()[i]->objects()[0];
        EXPECT_EQ(expectNumber[i], obj->children().size()) << "i = " << i << endl;
    }


    Box expectBox[] = {
            Box(583,456,417,145),
            Box(409, 517, 392, 129),
            Box(484, 532, 504, 164),
            Box(166, 512, 462, 150),
            Box(416, 434, 516, 152),
            Box(626, 416, 456, 160),
            Box(122, 412, 410, 164),
            Box(838, 396, 486, 182),
            Box(752, 448, 450, 176)
    };

    for (int i = 0; i < 9; ++i) {
        vector<Object *>v = fb->frames()[i]->objects()[0]->children();

        for (int j = 0; j < v.size(); ++j) {
            Vehicle *vehicle = (Vehicle*)v[j];
            EXPECT_LE(expectBox[i].x, vehicle->detection().box.x);
            EXPECT_LE(expectBox[i].y, vehicle->detection().box.y);
            EXPECT_GE(expectBox[i].width, vehicle->detection().box.width);
            EXPECT_GE(expectBox[i].height, vehicle->detection().box.height);
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

#endif
