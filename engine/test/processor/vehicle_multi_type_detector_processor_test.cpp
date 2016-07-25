#if true

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
    /**
    if (head) {
        delete head;
        head = NULL;
    }
     **/
}

static Operation getOperation() {
    Operation op;
    op.Set(OPERATION_VEHICLE |
           OPERATION_VEHICLE_DETECT );
    return op;
}

TEST(VehicleMultiTypeDectorTest, vehicleTypeTest) {
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

    destory();
}

TEST(VehicleMultiTypeDectorTest, vehicleNumberTest) {
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
    destory();
}

TEST(VehicleMultiTypeDectorTest, strangeInputTest) {
    init();
    FrameBatch *fb = fbhelper->getFrameBatch();
    head->process(fb);

    EXPECT_EQ(0, fb->batch_size());

    fbhelper->setBasePath("data/testimg/vehicleMultiType/strangeInput/");
    fbhelper->readImage(getOperation());

    fb = fbhelper->getFrameBatch();
    cout << fb->batch_size() << endl;

    head->process(fb);
    EXPECT_EQ(3, fb->batch_size());

    cv::Mat mat1(0, 1, 0);
    Frame *f1 = new Frame(1001, mat1);
    f1->set_operation(getOperation());
    fb->AddFrame(f1);
    head->getProcessor()->Update(f1);
    EXPECT_EQ(0, f1->objects().size());


    cv::Mat mat2(0, 0, 1);
    Frame *f2 = new Frame(1002, mat2);
    fb->AddFrame(f2);

    cv::Mat mat3(1, 2, 3);
    Frame *f3 = new Frame(1003, mat3);
    vector<Rect> rois;
    Rect rect(1, 1, 0, 0);
    rois.push_back(rect);
    f3->set_roi(rois);
    fb->AddFrame(f3);

    head->process(fb);
    EXPECT_EQ(6, fb->batch_size());

    destory();
}


TEST(VehicleMultiTypeDectorTest, carOnlyTest) {
    VehicleCaffeDetectorConfig config;
    config.car_only = true;
    config.is_model_encrypt = false;
    config.target_max_size = 600;
    config.target_min_size = 400;
    config.deploy_file = "data/models/310.txt";
    config.model_file = "data/models/310.dat";
    config.confirm_deploy_file = "data/models/311.txt";
    config.confirm_model_file = "data/models/311.dat";
    VehicleMultiTypeDetectorProcessor *carOnlyProcessor =
            new VehicleMultiTypeDetectorProcessor(config);


    fbhelper = new FrameBatchHelper(1);
    fbhelper->setBasePath("data/testimg/vehicleMultiType/carOnly/");
    fbhelper->readImage(getOperation());

    FrameBatch *fb = fbhelper->getFrameBatch();

    carOnlyProcessor->Update(fb);

    for (int i = 0; i < fb->batch_size(); ++i) {
        EXPECT_EQ(OBJECT_CAR, fb->frames()[i]->objects()[0]->type());
    }

    //delete carOnlyProcessor;
}

#endif
