#if true

#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "processor/face_detect_processor.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static FaceDetectProcessor *fdprocessor;

static void initConfig() {
    FaceDetector::FaceDetectorConfig config;
    config.is_model_encrypt = false;
    config.deploy_file = "data/models/400.txt";
    config.model_file = "data/models/400.dat";
    fdprocessor = new FaceDetectProcessor(config);
    fbhelper = new FrameBatchHelper(1);
}

static void destory() {
    if (fbhelper) {
        delete fbhelper;
        fbhelper = NULL;
    }
    if (fdprocessor) {
        delete fdprocessor;
        fdprocessor = NULL;
    }
}

static Operation getOperation() {
    Operation op;
    op.Set( OPERATION_FACE |
            OPERATION_FACE_DETECTOR);
    return op;
}

TEST(FaceDetectProcessorTest, faceDetectTest) {
    initConfig();
    fbhelper->setBasePath("data/testimg/face/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();

    fdprocessor->Update(fb);
    for (int i = 0; i < 7; ++i) {
        EXPECT_EQ(OBJECT_FACE, fb->frames()[i]->objects()[0]->type()) << "i = " << i;
    }
    EXPECT_EQ(0, fb->frames()[7]->objects().size());
    EXPECT_EQ(0, fb->frames()[8]->objects().size());

    delete fbhelper;
    fbhelper = new FrameBatchHelper(1);
    fbhelper->setBasePath("data/testimg/face/");
    fbhelper->readImage(getOperation());
    fb = fbhelper->getFrameBatch();

    for (int i = 0; i < 7; ++i) {
        fdprocessor->Update(fb->frames()[i]);
        EXPECT_EQ(OBJECT_FACE, fb->frames()[i]->objects()[0]->type()) << "i = " << i;
    }
    fdprocessor->Update(fb->frames()[7]);
    EXPECT_EQ(0, fb->frames()[7]->objects().size());
    fdprocessor->Update(fb->frames()[8]);
    EXPECT_EQ(0, fb->frames()[8]->objects().size());

    destory();
}

TEST(FaceDetectProcessorTest, strangeInputTest) {
    initConfig();
    fbhelper->setBasePath("data/testimg/face/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();

    cv::Mat mat1(0, 1, 0);
    Frame *f1 = new Frame(1001, mat1);
    f1->set_operation(getOperation());
    fdprocessor->Update(f1);
    EXPECT_EQ(0, f1->objects().size());
    fb->AddFrame(f1);


    cv::Mat mat2(1, 1, 1);
    Frame *f2 = new Frame(1002, mat2);
    fdprocessor->Update(f2);
    EXPECT_EQ(0, f2->objects().size());
    fb->AddFrame(f2);

    cv::Mat mat3(1, 2, 3);
    Frame *f3 = new Frame(1003, mat3);
    fdprocessor->Update(f3);
    EXPECT_EQ(0, f3->objects().size());
    fb->AddFrame(f3);

    fdprocessor->Update(fb);

    destory();
}

#endif
