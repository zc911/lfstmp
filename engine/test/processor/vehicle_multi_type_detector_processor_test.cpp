#include <algorithm>
#include "gtest/gtest.h"

#include "frame_batch_helper.h"
#include "vehicle_processor_head.h"
#include "file_reader.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static VehicleProcessorHead *head;
static FileReader *resultReader;

static void init() {
    head = new VehicleProcessorHead();
    fbhelper = new FrameBatchHelper(1);
    resultReader = NULL;
}

static void destory() {
    if (fbhelper) {
        delete fbhelper;
        fbhelper = NULL;
    }
    if (resultReader) {
        delete resultReader;
        resultReader = NULL;
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

TEST(VehicleMultiTypeDectorTest, vehicleTypeTest) {
    init();
    fbhelper->setBasePath("data/testimg/vehicleMultiType/type/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();

    resultReader = new FileReader("data/testimg/vehicleMultiType/type/result.txt");
    EXPECT_TRUE(resultReader->is_open());
    resultReader->read(",");
    head->process(fb);

    for (int i = 0; i < fb->batch_size(); ++i) {
        vector<int> expectResult, realResult;
        for (int j = 0; j < fb->frames()[i]->objects().size(); ++j) {
            realResult.push_back(fb->frames()[i]->objects()[j]->type());
        }
        stringstream s;
        s << i;
        for (int j = 0; j < resultReader->getValue(s.str()).size(); ++j) {
            expectResult.push_back(resultReader->getIntValue(s.str(), j));
        }
        sort(expectResult.begin(), expectResult.end());
        sort(realResult.begin(), realResult.end());
        EXPECT_EQ(realResult.size(), expectResult.size()) << "i = " << i << endl;
        for (int j = 0; j < expectResult.size(); ++j) {
            EXPECT_EQ(realResult[j], expectResult[j])  << "i = " << i << " j = " << j << endl;
        }
    }

    destory();
}

TEST(VehicleMultiTypeDectorTest, vehicleNumberTest) {
    init();
    fbhelper->setBasePath("data/testimg/vehicleMultiType/number/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();
    resultReader = new FileReader("data/testimg/vehicleMultiType/number/result.txt");
    EXPECT_TRUE(resultReader->is_open());
    resultReader->read(",");

    head->process(fb);

    for (int i = 0; i < fb->batch_size(); ++i) {
        stringstream s;
        s << i;
        EXPECT_EQ(resultReader->getIntValue(s.str(), 0), fb->frames()[i]->get_object_size());
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

    resultReader = new FileReader("data/testimg/vehicleMultiType/strangeInput/result.txt");
    EXPECT_TRUE(resultReader->is_open());
    resultReader->read(",");

    fb = fbhelper->getFrameBatch();
    head->process(fb);

    for (int i = 0; i < fb->batch_size(); ++i) {
        stringstream s;
        s << i;
        EXPECT_EQ(resultReader->getValue(s.str()).size(), fb->frames()[i]->objects().size()) << "i = " << i << endl;
    }

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
    string baseModelPath;
#ifdef UNENCRYPTMODEL
    config.is_model_encrypt = false;
    baseModelPath = "data/0/";
#else
    config.is_model_encrypt = true;
    baseModelPath = "data/1/";
#endif
    config.car_only = true;
    config.target_max_size = 600;
    config.target_min_size = 400;
    config.deploy_file = baseModelPath + "310.txt";
    config.model_file = baseModelPath + "310.dat";
    config.confirm_deploy_file = baseModelPath + "311.txt";
    config.confirm_model_file = baseModelPath + "311.dat";
    VehicleMultiTypeDetectorProcessor *carOnlyProcessor =
            new VehicleMultiTypeDetectorProcessor(config);


    fbhelper = new FrameBatchHelper(1);
    fbhelper->setBasePath("data/testimg/vehicleMultiType/carOnly/");
    fbhelper->readImage(getOperation());

    resultReader = new FileReader("data/testimg/vehicleMultiType/carOnly/result.txt");
    EXPECT_TRUE(resultReader->is_open());
    resultReader->read(",");

    FrameBatch *fb = fbhelper->getFrameBatch();

    carOnlyProcessor->Update(fb);

    for (int i = 0; i < fb->batch_size(); ++i) {
        stringstream s;
        s << i;
        EXPECT_EQ(resultReader->getIntValue(s.str(), 0), fb->frames()[i]->objects()[0]->type());
    }

    delete carOnlyProcessor;
}

TEST(VehicleMultiTypeDectorTest, vehiclePositionTest) {
    init();
    fbhelper->setBasePath("data/testimg/vehicleMultiType/objectPosition/");
    fbhelper->readImage(getOperation());
    FrameBatch *fb = fbhelper->getFrameBatch();

    resultReader = new FileReader("data/testimg/vehicleMultiType/objectPosition/result.txt");
    EXPECT_TRUE(resultReader->is_open());
    resultReader->read(",");
    head->process(fb);

    for (int i = 0; i < fb->batch_size(); ++i) {
        vector<int> expectResult, realResult;
        for (int j = 0; j < fb->frames()[i]->objects().size(); ++j) {
            realResult.push_back(fb->frames()[i]->objects()[j]->detection().box.x);
            realResult.push_back(fb->frames()[i]->objects()[j]->detection().box.y);
            realResult.push_back(fb->frames()[i]->objects()[j]->detection().box.height);
            realResult.push_back(fb->frames()[i]->objects()[j]->detection().box.width);
        }
        stringstream s;
        s << i;
        for (int j = 0; j < resultReader->getValue(s.str()).size(); ++j) {
            expectResult.push_back(resultReader->getIntValue(s.str(), j));
        }
        int totalArea = 0;
        for (int j = 0; j < realResult.size() / 4; ++j) {
            totalArea += realResult[j * 4 + 2] * realResult[j * 4 + 3];
        }
        EXPECT_LT(0, totalArea);

        int realArea = 0;
        for (int j = 0; j < realResult.size() / 4; ++j) {
            int x = realResult[j * 4];
            int y = realResult[j * 4 + 1];
            int height = realResult[j * 4 + 2];
            int width = realResult[j * 4 + 3];

            for (int k = 0; k < expectResult.size() / 4; ++k) {
                int ex = expectResult[k * 4];
                int ey = expectResult[k * 4 + 1];
                int eheight = expectResult[k * 4 + 2];
                int ewidth = expectResult[k * 4 + 3];
                if (x >= ex && x + width <= ex + ewidth && y >= ey && y + height <= ey + eheight) {
                    realArea += height * width;
                }
                else if (x <= ex && x + width >= ex + ewidth && y <= ey && y + height >= ey + eheight) {
                    realArea += eheight * ewidth;
                }
                else if (x >= ex && x <= ex + ewidth && y >= ey && y <= ey + eheight) {
                    realArea += min(ex + ewidth - x, width) * min(ey + eheight - y, height);
                }
                else if (y + height >= ey && y + height <= ey + eheight && x >= ex && x <= ex + ewidth) {
                    realArea += (y + height - ey) * min(width, x + width - ex - ewidth);
                }
                else if (x + width >= ex && x + width <= ex + ewidth && y >= ey && y <= ey + eheight) {
                    realArea += (x + width - ex) * min(height, y + height - ey - eheight);
                }
                else if (x + width >= ex && x + width <= ex + ewidth && y + height >= ey && y + height <= ey + eheight) {
                    realArea += (x + width - ex) * (y + height - ey);
                }
            }
        }
        EXPECT_LE(60, realArea * 100 / totalArea) << "i = " << i << endl;
    }

    destory();
}
