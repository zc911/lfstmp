#if false

#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "vehicle_processor_head.h"
#include "processor/pedestrian_classifier_processor.h"
#include "file_reader.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static VehicleProcessorHead *head;
static PedestrianClassifierProcessor *pcprocessor;
static FileReader *resultReader;

static void initConfig() {
    PedestrianClassifier::PedestrianConfig config;
    config.is_model_encrypt = false;
    config.deploy_file = "data/models/1000.txt";
    config.model_file = "data/models/1000.dat";
    config.tag_name_path = "data/models/pedestrian_attribute_tagnames.txt";
    config.layer_name = "loss3/classifier_personattrib_47";
    pcprocessor = new PedestrianClassifierProcessor(config);
}

static void init() {
    resultReader = NULL;
    initConfig();
    head = new VehicleProcessorHead();
    fbhelper = new FrameBatchHelper(1);
    head->setNextProcessor(pcprocessor);
}

static void destory() {
    /**
    if (head) {
        delete head;
        head = NULL;
    }
     **/

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
            OPERATION_VEHICLE_PEDESTRIAN_ATTR |
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
        EXPECT_EQ(resultReader->getIntValue(s.str(), 0), total);
    }

    destory();
}

TEST(PedestrianClassiFierProcessorTest, pedestrianAttributeTest) {
    init();
    fbhelper->setBasePath("data/testimg/pedestrian/attribute/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();
    head->process(fb);
    FileReader mapping("data/models/pedestrian_attribute_tagnames.txt");
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
            }
        }
        EXPECT_EQ(resultReader->getIntValue(s.str(), 0), total);
    }

    for (int i = 0; i < fb->frames().size(); ++i) {

    }
    /**
    Pedestrian *  p = (Pedestrian *)fb->frames()[0]->objects()[0];
    EXPECT_LE(0.6, p->attrs()[1].confidence);
    EXPECT_LE(0.6, p->attrs()[24].confidence);
    EXPECT_LE(0.6, p->attrs()[35].confidence);
    EXPECT_LE(0.6, p->attrs()[40].confidence);
    EXPECT_GE(0.4, p->attrs()[45].confidence);

    p = (Pedestrian *)fb->frames()[1]->objects()[0];
    EXPECT_LE(0.6, p->attrs()[10].confidence);
    EXPECT_LE(0.6, p->attrs()[24].confidence);
    EXPECT_LE(0.6, p->attrs()[36].confidence);
    EXPECT_LE(0.6, p->attrs()[40].confidence);
    EXPECT_LE(0.6, p->attrs()[44].confidence);

    p = (Pedestrian *)fb->frames()[2]->objects()[0];
    EXPECT_LE(0.6, p->attrs()[12].confidence);
    EXPECT_LE(0.6, p->attrs()[24].confidence);
    EXPECT_LE(0.6, p->attrs()[36].confidence);
    EXPECT_LE(0.6, p->attrs()[40].confidence);
    EXPECT_LE(0.6, p->attrs()[44].confidence);
    EXPECT_LE(0.6, p->attrs()[45].confidence);

    p = (Pedestrian *)fb->frames()[3]->objects()[0];
    EXPECT_LE(0.6, p->attrs()[12].confidence);
    EXPECT_LE(0.6, p->attrs()[24].confidence);
    EXPECT_LE(0.6, p->attrs()[40].confidence);
    EXPECT_LE(0.6, p->attrs()[44].confidence);
    EXPECT_GE(0.4, p->attrs()[45].confidence);

    p = (Pedestrian *)fb->frames()[4]->objects()[0];
    EXPECT_LE(0.6, p->attrs()[10].confidence);
    EXPECT_LE(0.6, p->attrs()[12].confidence);
    EXPECT_LE(0.6, p->attrs()[35].confidence);
    EXPECT_LE(0.6, p->attrs()[44].confidence);
    EXPECT_LE(0.6, p->attrs()[45].confidence);

    p = (Pedestrian *)fb->frames()[5]->objects()[0];
    EXPECT_LE(0.6, p->attrs()[35].confidence);
    EXPECT_LE(0.6, p->attrs()[40].confidence);
    EXPECT_LE(0.6, p->attrs()[44].confidence);

    p = (Pedestrian *)fb->frames()[6]->objects()[0];
    EXPECT_LE(0.6, p->attrs()[10].confidence);
    EXPECT_LE(0.6, p->attrs()[26].confidence);
    EXPECT_LE(0.6, p->attrs()[40].confidence);
    EXPECT_LE(0.6, p->attrs()[44].confidence);
    EXPECT_LE(0.6, p->attrs()[45].confidence);

    delete fbhelper;
    fbhelper = new FrameBatchHelper(1);
    fbhelper->setBasePath("data/testimg/pedestrian/attribute/");
    fbhelper->readImage(getOperation());
    fb = fbhelper->getFrameBatch();

    for (int i = 0; i < fb->batch_size(); ++i) {
        head->getProcessor()->Update(fb->frames()[i]);
        EXPECT_EQ(0, fb->frames()[i]->objects().size());
    }
     **/

    destory();
}

#endif
