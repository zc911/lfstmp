#if true

#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "processor/face_detect_processor.h"
#include "processor/face_feature_extract_processor.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static FaceDetectProcessor *fdprocessor;
static FaceFeatureExtractProcessor *ffeprocessor;

static void initConfig() {
    FaceDetector::FaceDetectorConfig dConfig;
    dConfig.is_model_encrypt = false;
    dConfig.deploy_file = "data/models/400.txt";
    dConfig.model_file = "data/models/400.dat";
    fdprocessor = new FaceDetectProcessor(dConfig);

    FaceFeatureExtractor::FaceFeatureExtractorConfig fConfig;
    fConfig.is_model_encrypt = false;
    fConfig.deploy_file = "data/models/500.txt";
    fConfig.model_file = "data/models/500.dat";
    fConfig.align_deploy = "data/models/avgface.jpg";
    fConfig.align_model = "data/models/501.dat";

    ffeprocessor  = new FaceFeatureExtractProcessor(fConfig);

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
            OPERATION_FACE_FEATURE_VECTOR |
            OPERATION_FACE_DETECTOR);
    return op;
}

TEST(FaceFeatureExtractProcessorTest, faceFeatureExtractTest) {
    initConfig();
    fbhelper->setBasePath("data/testimg/face/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();
    fdprocessor->SetNextProcessor(ffeprocessor);

    fdprocessor->Update(fb);
    for (int i = 0; i < 7; ++i) {
        Face *f = (Face*)fb->frames()[i]->objects()[0];
        EXPECT_LE(256, f->feature().Serialize().size());
    }
    EXPECT_EQ(0, fb->frames()[7]->objects().size());
    EXPECT_EQ(0, fb->frames()[8]->objects().size());
    Object *obj1 = new Object(OBJECT_CAR);
    Object *obj2 = new Object(OBJECT_UNKNOWN);
    fb->frames()[7]->put_object(obj1);
    fb->frames()[8]->put_object(obj2);
    fdprocessor->Update(fb);
    EXPECT_EQ(OBJECT_CAR, obj1->type());
    EXPECT_EQ(OBJECT_UNKNOWN, obj2->type());

    delete fbhelper;
    fbhelper = new FrameBatchHelper(1);
    fbhelper->setBasePath("data/testimg/face/");
    fbhelper->readImage(getOperation());
    fb = fbhelper->getFrameBatch();

    for (int i = 0; i < 7; ++i) {
        fdprocessor->Update(fb->frames()[i]);
        Face *f = (Face*)fb->frames()[i]->objects()[0];
        EXPECT_LE(256, f->feature().Serialize().size());
    }
    obj1 = new Object(OBJECT_CAR);
    fb->frames()[7]->put_object(obj1);
    fdprocessor->Update(fb->frames()[7]);
    EXPECT_EQ(OBJECT_CAR, obj1->type());
    obj2 = new Object(OBJECT_UNKNOWN);
    fb->frames()[8]->put_object(obj2);
    EXPECT_EQ(OBJECT_UNKNOWN, obj2->type());

    destory();
}

#endif
