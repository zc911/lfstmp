#if true

#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "processor/car_rank_processor.h"
#include "file_reader.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static CarRankProcessor *crprocessor;
static FileReader *resultReader;
static vector<cv::Mat *> vMat;
static vector<vector<Rect> *> vHotspots;
static vector<vector<CarRankFeature> *> vCandidates;

static void initConfig() {
    Config config;
    //config.Load("data/config.json");
    crprocessor = new CarRankProcessor(config);
}

static void init() {
    initConfig();
    fbhelper = new FrameBatchHelper(1);
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
    if (crprocessor) {
        delete crprocessor;
        crprocessor = NULL;
    }
    for (int i = 0; i < vMat.size(); ++i) {
        delete vMat[i];
    }
    vMat.clear();
    for (int i = 0; i < vHotspots.size(); ++i) {
        delete vHotspots[i];
    }
    vHotspots.clear();
    for (int i = 0; i < vCandidates.size(); ++i) {
        delete vCandidates[i];
    }
    vCandidates.clear();
}

static Operation getOperation() {
    Operation op;
    op.Set( OPERATION_VEHICLE |
            OPERATION_VEHICLE_DETECT );
    return op;
}

bool readImg(string basePath, int index) {
    stringstream s;
    s << index;
    string imageName = basePath + s.str() + ".jpg";
    cv::Mat tmpMat = cv::imread(imageName.c_str());
    if (tmpMat.empty()) {
        return false;
    }

    cv::Mat *image = new cv::Mat(tmpMat);
    vMat.push_back(image);

    vector<Rect> *hotspots = new vector<Rect>();
    vHotspots.push_back(hotspots);
    int x = resultReader->getIntValue(s.str(), 2);
    int y = resultReader->getIntValue(s.str(), 3);
    int width = resultReader->getIntValue(s.str(), 4);
    int height = resultReader->getIntValue(s.str(), 5);

    Rect tmpRect(x, y, width, height);
    hotspots->push_back(tmpRect);

    int candidateNumber = resultReader->getIntValue(s.str(), 0);

    vector<CarRankFeature> *candidates = new vector<CarRankFeature>();
    vCandidates.push_back(candidates);
    string baseCandidate = basePath + s.str() + "_";
    for (int i = 0; i < candidateNumber; ++i) {
        ifstream file;
        s.str("");
        s << i;
        file.open(baseCandidate + s.str() + ".candidate");
        EXPECT_TRUE(file.is_open());
        string carFeature;
        while (!file.eof()) {
            char ch = file.get();
            if (ch == -1) break;
            carFeature += ch;
        }

        CarRankFeature carRankFeature;
        EXPECT_TRUE(carRankFeature.Deserialize(carFeature));
        candidates->push_back(carRankFeature);
    }
    CarRankFrame *frame = new CarRankFrame(index, *image, *hotspots, *candidates);
    frame->set_operation(getOperation());
    fbhelper->getFrameBatch()->AddFrame(frame);
    return true;
}

TEST(CarRankProcessorTest, carRankTest) {
    init();
    resultReader = new FileReader("data/testimg/carRank/result.txt");
    EXPECT_TRUE(resultReader->is_open());
    resultReader->read(",");


    for (int i = 0; ; ++i) {
        if (readImg("data/testimg/carRank/", i) == false) {
            break;
        }
    }
    FrameBatch *fb = fbhelper->getFrameBatch();
    crprocessor->Update(fb);
    for (int i = 0; i < fb->batch_size(); ++i) {
        EXPECT_EQ(0, fb->frames()[i]->objects().size());
    }
    for (int i = 0; i < fb->batch_size(); ++i) {
        CarRankFrame *frame = (CarRankFrame *)(fb->frames()[i]);
        crprocessor->Update(frame);
        int idx = 0;
        for (int j = 0; j < frame->result_.size(); ++j) {
            if (frame->result_[j].score_ > frame->result_[idx].score_) {
                idx = j;
            }
        }
        stringstream s;
        s << i;
        EXPECT_EQ(resultReader->getIntValue(s.str(), 1), frame->result_[idx].index_);
    }

    destory();
}

#endif
