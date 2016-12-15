//#include <alg/classification/pedestrian_classifier.h>
#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "vehicle_processor_head.h"
#include "processor/pedestrian_classifier_processor.h"
#include "file_reader.h"
#include "algorithm_factory.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static VehicleProcessorHead *head;
static PedestrianClassifierProcessor *pcprocessor;
static FileReader *resultReader;

static void initConfig() {
    dgvehicle::AlgorithmFactory::GetInstance()->Initialize("data/dgvehicle", 0, false);
    pcprocessor = new PedestrianClassifierProcessor();
}

static void init() {
    resultReader = NULL;
    initConfig();
    head = new VehicleProcessorHead();
    fbhelper = new FrameBatchHelper(1);
    head->setNextProcessor(pcprocessor);
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
            OPERATION_PEDESTRIAN_ATTR |
            OPERATION_VEHICLE_DETECT );
    return op;
}

TEST(PedestrianClassiFierProcessorTest, pedestrianNumberTest) {
    init();
    fbhelper->setBasePath("data/testimg/pedestrian/number/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();
    head->process(fb);

    resultReader = new FileReader("data/testimg/pedestrian/number/result.txt");
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
            if (fb->frames()[i]->objects()[j]->type() == OBJECT_PEDESTRIAN) {
                ++total;
            }
        }
        EXPECT_EQ(resultReader->getIntValue(s.str(), 0), total) << "i = " << i << endl;
    }

    destory();
}

TEST(PedestrianClassiFierProcessorTest, pedestrianAttributeTest) {
    init();
    fbhelper->setBasePath("data/testimg/pedestrian/attribute/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();
    head->process(fb);

#ifdef UNENCRYPTMODEL
    FileReader mapping("data/dgvehicle/model/classifier/Pedestrian/pedestrian_attribute_tagnames.txt");
#else
    FileReader mapping("data/dgvehicle/encryptModel/classifier/Pedestrian/pedestrian_attribute_tagnames.txt");
#endif

    EXPECT_TRUE(mapping.is_open());
    mapping.read(" ");
    cout << mapping.getValue("head_hat")[0] << endl;

    resultReader = new FileReader("data/testimg/pedestrian/attribute/result.txt");
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
            if (fb->frames()[i]->objects()[j]->type() == OBJECT_PEDESTRIAN) {
                ++total;
                Pedestrian * p = (Pedestrian *)fb->frames()[i]->objects()[j];
                int attrCnt = 0;
                for (int i = 0; i < p->attrs().size(); ++i) {
                    if (p->attrs()[i].confidence > 0.01) {
                        ++attrCnt;
                    }
                }
                EXPECT_LE(2, attrCnt);
            }
        }
        EXPECT_EQ(resultReader->getIntValue(s.str(), 0), total) << "i = " << i << endl;
    }
    delete fbhelper;

    fbhelper = new FrameBatchHelper(1);
    fbhelper->setBasePath("data/testimg/pedestrian/attribute/");
    fbhelper->readImage(getOperation());
    fb = fbhelper->getFrameBatch();

    for (int i = 0; i < fb->batch_size(); ++i) {
        head->getProcessor()->Update(fb->frames()[i]);
        EXPECT_EQ(0, fb->frames()[i]->objects().size());
    }

    destory();
}
