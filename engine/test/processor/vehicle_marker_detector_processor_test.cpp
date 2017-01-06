#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "vehicle_processor_head.h"
#include "processor/vehicle_marker_classifier_processor.h"
#include "file_reader.h"
#include "algorithm_factory.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static VehicleProcessorHead *head;
static VehicleMarkerClassifierProcessor *vmcprocessor;
static FileReader *resultReader;
static VehicleWindowProcessor *window;
static void initConfig() {
    dgvehicle::AlgorithmFactory::GetInstance()->Initialize("data/dgvehicle", 0, false);

    vmcprocessor = new VehicleMarkerClassifierProcessor(false);
}

static Operation getOperation() {
    Operation op;
    op.Set(OPERATION_VEHICLE |
        OPERATION_VEHICLE_MARKER |
        OPERATION_VEHICLE_DETECT);
    return op;
}

static void init() {
    resultReader = NULL;
    head = new VehicleProcessorHead();

    fbhelper = new FrameBatchHelper(1);

    window = new VehicleWindowProcessor();
    initConfig();

    head->setNextProcessor(window->getProcessor());
    window->setNextProcessor(vmcprocessor);
    dgvehicle::AlgorithmFactory::GetInstance()->ReleaseUselessModel();

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

TEST(VehicleMarkerDetectorTest, markerDetectorTest) {
    init();
    fbhelper->setBasePath("data/testimg/markerDetector/");
    fbhelper->readImage(getOperation());
    head->process(fbhelper->getFrameBatch());
    FrameBatch *fb = fbhelper->getFrameBatch();
    resultReader = new FileReader("data/testimg/markerDetector/result.txt");

    EXPECT_TRUE(resultReader->is_open());
    resultReader->read(",");

    for (int i = 0; i < fb->batch_size(); ++i) {
        vector<Object *> vehicles = fb->frames()[i]->objects();

        stringstream s;
        s << i;
        int symbols[10];
        for (int j = 0; j < 10; j++)
            symbols[j] = 0;

        for (int j = 0; j < vehicles.size(); j++) {
            Vehicle *obj = (Vehicle *) vehicles[j];
            Window *w = (Window *) obj->child(OBJECT_WINDOW);

            if (w == NULL)
                continue;

            vector<Object *> ms = w->children();

            for (int k = 0; k < ms.size(); k++) {
                Marker *m = (Marker *) ms[k];
                EXPECT_LE(0, w->detection().box().x);
                EXPECT_LE(0, w->detection().box().y);
                EXPECT_GE(obj->detection().box().width, m->detection().box().width + m->detection().box().x)
                                << "i = " << i << endl;
                EXPECT_GE(obj->detection().box().height, m->detection().box().height + m->detection().box().y)
                                << "i = " << i << endl;
                symbols[m->class_id()]++;
            }
        }
        for (int j = 0; j < 10; ++j) {
            EXPECT_EQ(resultReader->getIntValue(s.str(), j), symbols[j]) << "i = " << i << " j = " << j << endl;
        }
    }

    destory();
}

