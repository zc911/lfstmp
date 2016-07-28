#if false

#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "processor/face_detect_processor.h"
#include "processor/face_feature_extract_processor.h"
#include "file_reader.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static FaceDetectProcessor *fdprocessor;
static FaceFeatureExtractProcessor *ffeprocessor;
static FileReader *resultReader;

static void initConfig() {
    resultReader = NULL;
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
    if (resultReader) {
        delete resultReader;
        resultReader = NULL;
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
    fbhelper->setBasePath("data/testimg/face/featureExtract/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();
    fdprocessor->SetNextProcessor(ffeprocessor);

    resultReader = new FileReader("data/testimg/face/featureExtract/result.txt");
    EXPECT_TRUE(resultReader->is_open());
    resultReader->read(",");

    fdprocessor->Update(fb);
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
            if (fb->frames()[i]->objects()[j]->type() == OBJECT_FACE) {
                ++total;
                Face *f = (Face *) fb->frames()[i]->objects()[j];
                EXPECT_LE(256, f->feature().Serialize().size());
            }
        }
        EXPECT_EQ(resultReader->getIntValue(s.str(), 0), total);
    }

    destory();
}

#endif
