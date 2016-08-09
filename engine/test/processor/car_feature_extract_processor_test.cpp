#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "vehicle_processor_head.h"
#include "processor/car_feature_extract_processor.h"
#include "file_reader.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static VehicleProcessorHead *head;
static CarFeatureExtractProcessor *cfeprocessor;
static FileReader *resultReader;

static void init() {
    resultReader = NULL;
    cfeprocessor = new CarFeatureExtractProcessor();
    head = new VehicleProcessorHead();
    fbhelper = new FrameBatchHelper(1);
    head->setNextProcessor(cfeprocessor);
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
    if (resultReader) {
        delete resultReader;
        resultReader = NULL;
    }
}

static Operation getOperation() {
    Operation op;
    op.Set( OPERATION_VEHICLE |
            OPERATION_VEHICLE_FEATURE_VECTOR |
            OPERATION_VEHICLE_DETECT );
    return op;
}

TEST(CarFeatureExtractProcessorTest, carFeatureExtractTest) {
    init();
    fbhelper->setBasePath("data/testimg/carFeatureExtract/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();
    head->process(fb);

    resultReader = new FileReader("data/testimg/carFeatureExtract/result.txt");
    EXPECT_TRUE(resultReader->is_open());
    resultReader->read(",");

    for (int i = 0; i < fb->batch_size(); ++i) {
        stringstream s;
        s << i;
        if (resultReader->getIntValue(s.str(), 0) == 0) {
            if (fb->frames()[i]->objects().empty()) {
                continue;
            }
        }
        int total = 0;
        for (int j = 0; j < fb->frames()[i]->objects().size(); ++j) {

            Object *obj = fb->frames()[i]->objects()[j];
            if (obj->type() == OBJECT_CAR) {
                ++total;
                Vehicle *v = (Vehicle *)obj;
                EXPECT_LT(256, v->feature().Serialize().size());
            }
        }
        EXPECT_EQ(resultReader->getIntValue(s.str(), 0), total);
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
