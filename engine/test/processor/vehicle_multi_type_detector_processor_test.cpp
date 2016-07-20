#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "vehicle_processor_head.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static VehicleProcessorHead *head;

static void init() {
    head = new VehicleProcessorHead();
    fbhelper = new FrameBatchHelper(1);
}

static void destory() {
    if (fbhelper) {
        delete fbhelper;
        fbhelper = NULL;
    }
    if (head) {
        delete head;
        head = NULL;
    }
}

static Operation getOperation() {
    Operation op;
    op.Set(OPERATION_VEHICLE |
           OPERATION_VEHICLE_DETECT );
    return op;
}

TEST(VehicleMultiTypeDectorTest, VehicleTypeTest) {
    init();
    fbhelper->setBasePath("data/testimg/vehicleMultiType/type/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();
    head->process(fb);

    ObjectType expectType[] = {
            OBJECT_CAR,
            OBJECT_CAR,
            OBJECT_CAR,
            OBJECT_BICYCLE,
            OBJECT_BICYCLE,
            OBJECT_BICYCLE,
            OBJECT_BICYCLE,
            OBJECT_BICYCLE,
            OBJECT_TRICYCLE,
            OBJECT_TRICYCLE
    };

    for (int i = 0; i < fb->batch_size(); ++i) {
        Object *obj = fb->frames()[i]->objects()[0];
        EXPECT_EQ(expectType[i], obj->type()) << "i = " << i << endl;
    }

    fbhelper->printFrame();
    //destory();
}

TEST(VehicleMultiTypeDectorTest, VehicleNumberTest) {
    init();
    fbhelper->setBasePath("data/testimg/vehicleMultiType/number/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();
    head->process(fb);

    int expectNum[] = {
            3, 1, 0, 0, 5, 2, 2, 3, 2
    };

    for (int i = 0; i < fb->batch_size(); ++i) {
        EXPECT_EQ(expectNum[i], fb->frames()[i]->get_object_size()) << "i = " << i << endl;
    }
    fbhelper->printFrame();
//    destory();
}

TEST(VehicleMultiTypeDectorTest, StrangeInputTest) {
    init();
    FrameBatch *fb = fbhelper->getFrameBatch();
    head->process(fb);

    EXPECT_EQ(0, fb->batch_size());

    fbhelper->setBasePath("data/testimg/vehicleMultiType/strangeInput/");
    int num = fbhelper->readImage(getOperation());

    fb = fbhelper->getFrameBatch();
    cout << fb->batch_size() << endl;

    head->process(fb);

    EXPECT_EQ(3, fb->batch_size());

//    destory();
}
