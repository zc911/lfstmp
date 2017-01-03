#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "vehicle_processor_head.h"
#include "processor/vehicle_window_detector_processor.h"
#include "file_reader.h"
#include "algorithm_factory.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static VehicleProcessorHead *head;
static VehicleWindowDetectorProcessor *vwdprocessor;
static FileReader *resultReader;

static void initConfig() {
    dgvehicle::AlgorithmFactory::GetInstance()->Initialize("data/dgvehicle", 0, false);
/*    VehicleCaffeDetectorConfig wConfig;
    string baseModelPath;
#ifdef UNENCRYPTMODEL
    wConfig.is_model_encrypt = false;
    baseModelPath = "data/0/";
#else
    wConfig.is_model_encrypt = true;
    baseModelPath = "data/1/";
#endif

    wConfig.deploy_file = baseModelPath + "701.txt";
    wConfig.model_file = baseModelPath + "701.dat";

    wConfig.target_max_size = 160;
    wConfig.target_min_size = 80; */
    vwdprocessor = new VehicleWindowDetectorProcessor();
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
    resultReader = NULL;
    head = new VehicleProcessorHead();
    fbhelper = new FrameBatchHelper(1);
    head->setNextProcessor(vwdprocessor);
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

TEST(VehicleWindowDetectorTest, windowDetectorTest) {
    init();
    fbhelper->setBasePath("data/testimg/windowDetector/");
    fbhelper->readImage(getOperation());
    head->process(fbhelper->getFrameBatch());
    FrameBatch *fb = fbhelper->getFrameBatch();
    resultReader = new FileReader("data/testimg/windowDetector/result.txt");

    EXPECT_TRUE(resultReader->is_open());
    resultReader->read(",");

    for (int i = 0; i < fb->batch_size(); ++i) {
        vector<Object *> vehicles = fb->frames()[i]->objects();
        int cnt = 0;
        stringstream s;
        s << i;

        for (int j = 0; j < vehicles.size(); j++) {
            Vehicle *obj = (Vehicle *)vehicles[j];
            Window *w = (Window *)obj->child(OBJECT_WINDOW);
            if (w == NULL)
                continue;

            EXPECT_LE(0, w->detection().box().x);
            EXPECT_LE(0, w->detection().box().y);
            EXPECT_GE(obj->detection().box().width, w->detection().box().width + w->detection().box().x);
            EXPECT_GE(obj->detection().box().height, w->detection().box().height + w->detection().box().y);
            cnt += obj->children(OBJECT_WINDOW).size();

        }
        EXPECT_EQ(resultReader->getIntValue(s.str(), 0), cnt ) << "i = " << i << endl;


    }

    destory();
}

