#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "processor/face_detect_processor.h"
#include "file_reader.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static FaceDetectProcessor *fdprocessor;
static FileReader *resultReader;

static void initConfig() {
    resultReader = NULL;
    FaceDetectorConfig config;
    string baseModelPath;
#ifdef UNENCRYPTMODEL
    config.is_model_encrypt = false;
    baseModelPath = "data/0/";
#else
    config.is_model_encrypt = true;
    baseModelPath = "data/1/";
#endif
    config.deploy_file = baseModelPath + "400.txt";
    config.model_file = baseModelPath + "400.dat";
    fdprocessor = new FaceDetectProcessor(config,1);
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
    if (resultReader) {
        delete resultReader;
        resultReader = NULL;
    }
}

static Operation getOperation() {
    Operation op;
    op.Set( OPERATION_FACE |
            OPERATION_FACE_DETECT);
    return op;
}


// TODO
TEST(FaceDetectProcessorTest, faceDetectTest) {
//    initConfig();
//    fbhelper->setBasePath("data/testimg/face/detect/");
//    fbhelper->readImage(getOperation());
//    FrameBatch *fb = fbhelper->getFrameBatch();
//
//    resultReader = new FileReader("data/testimg/face/detect/result.txt");
//    EXPECT_TRUE(resultReader->is_open());
//    resultReader->read(",");
//
//    fdprocessor->Update(fb);
//    for (int i = 0; i < fb->batch_size(); ++i) {
//        stringstream s;
//        s << i;
//        if (resultReader->getIntValue(s.str(), 0) == 0) {
//            if (fb->frames()[i]->objects().size() == 0) {
//                continue;
//            }
//        }
//        int total = 0;
//        for (int j = 0; j < fb->frames()[i]->objects().size(); ++j) {
//            if (fb->frames()[i]->objects()[j]->type() == OBJECT_FACE) {
//                ++total;
//            }
//        }
//        EXPECT_EQ(resultReader->getIntValue(s.str(), 0), total);
//    }
//
//    delete fbhelper;
//    fbhelper = new FrameBatchHelper(1);
//    fbhelper->setBasePath("data/testimg/face/detect/");
//    fbhelper->readImage(getOperation());
//    fb = fbhelper->getFrameBatch();
//    for (int i = 0; i < fb->batch_size(); ++i) {
//        fdprocessor->Update(fb->frames()[i]);
//        stringstream s;
//        s << i;
//        if (resultReader->getIntValue(s.str(), 0) == 0) {
//            if (fb->frames()[i]->objects().size() == 0) {
//                continue;
//            }
//        }
//        int total = 0;
//        for (int j = 0; j < fb->frames()[i]->objects().size(); ++j) {
//            if (fb->frames()[i]->objects()[j]->type() == OBJECT_FACE) {
//                ++total;
//            }
//        }
//        EXPECT_EQ(resultReader->getIntValue(s.str(), 0), total);
//    }
//
//    destory();
}

// TODO
TEST(FaceDetectProcessorTest, strangeInputTest) {
//    initConfig();
//    fbhelper->setBasePath("data/testimg/face/detect/");
//    fbhelper->readImage(getOperation());
//    FrameBatch *fb = fbhelper->getFrameBatch();
//
//    cv::Mat mat1(0, 1, 0);
//    Frame *f1 = new Frame(1001, mat1);
//    f1->set_operation(getOperation());
//    fdprocessor->Update(f1);
//    EXPECT_EQ(0, f1->objects().size());
//    fb->AddFrame(f1);
//
//
//    cv::Mat mat2(1, 1, 1);
//    Frame *f2 = new Frame(1002, mat2);
//    fdprocessor->Update(f2);
//    EXPECT_EQ(0, f2->objects().size());
//    fb->AddFrame(f2);
//
//    cv::Mat mat3(1, 2, 3);
//    Frame *f3 = new Frame(1003, mat3);
//    fdprocessor->Update(f3);
//    EXPECT_EQ(0, f3->objects().size());
//    fb->AddFrame(f3);
//
//    fdprocessor->Update(fb);
//
//    destory();
}
