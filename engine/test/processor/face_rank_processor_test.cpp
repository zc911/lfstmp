#include "gtest/gtest.h"
#include "frame_batch_helper.h"
#include "processor/face_rank_processor.h"
#include "file_reader.h"
#include "algorithm_factory.h"

using namespace std;
using namespace dg;

static FrameBatchHelper *fbhelper;
static FaceRankProcessor *frprocessor;
static FileReader *resultReader;
static vector<FaceRankFeature *> vFaceRankFeature;
static vector<vector<Rect> *> vHotspots;
static vector<vector<FaceRankFeature> *> vCandidates;

static void init() {
    dgvehicle::AlgorithmFactory::GetInstance()->Initialize("data/dgvehicle", 0, false);
    frprocessor = new FaceRankProcessor();
    fbhelper = new FrameBatchHelper(1);

    dgvehicle::AlgorithmFactory::GetInstance()->ReleaseUselessModel();
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
    if (frprocessor) {
        delete frprocessor;
        frprocessor = NULL;
    }
    for (int i = 0; i < vFaceRankFeature.size(); ++i) {
        delete vFaceRankFeature[i];
    }
    vFaceRankFeature.clear();
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

static string featureRead(string fileName) {
    string feature = "";
    ifstream file;
    file.open(fileName);
    if (file.is_open()) {
        while (!file.eof()) {
            char ch = file.get();
            if (ch == -1) break;
            feature += ch;
        }
    }
    return feature;
}

bool readFeature(string basePath, int index) {
    stringstream s;
    s << index;

    FaceRankFeature *faceRankFeature = new FaceRankFeature();
    string feature = featureRead(basePath + s.str() + ".feature");
    if (feature == "") {
        return false;
    }
    EXPECT_TRUE(faceRankFeature->Deserialize(feature));
    vFaceRankFeature.push_back(faceRankFeature);

    vector<Rect> *hotspots = new vector<Rect>();
    vHotspots.push_back(hotspots);
    int x = resultReader->getIntValue(s.str(), 2);
    int y = resultReader->getIntValue(s.str(), 3);
    int width = resultReader->getIntValue(s.str(), 4);
    int height = resultReader->getIntValue(s.str(), 5);

    Rect tmpRect(x, y, width, height);
    hotspots->push_back(tmpRect);

    int candidateNumber = resultReader->getIntValue(s.str(), 0);

    vector<FaceRankFeature> *candidates = new vector<FaceRankFeature>();
    vCandidates.push_back(candidates);
    string baseCandidate = basePath + s.str() + "_";
    for (int i = 0; i < candidateNumber; ++i) {
        s.str("");
        s << i;
        feature = featureRead(baseCandidate + s.str() + ".candidate");
        EXPECT_LT(0, feature.size());
        FaceRankFeature frFeature;
        EXPECT_TRUE(frFeature.Deserialize(feature));
        candidates->push_back(frFeature);
    }
    Mat img;
    FaceRankFrame *frame = new FaceRankFrame(index, img, *hotspots, *candidates);
    frame->set_operation(getOperation());
    fbhelper->getFrameBatch()->AddFrame(frame);
    frprocessor->Update(frame);
    return true;
}

TEST(FaceRankProcessorTest, faceRankTest) {
    init();
    resultReader = new FileReader("data/testimg/faceRank/result.txt");
    EXPECT_TRUE(resultReader->is_open());
    resultReader->read(",");

    for (int i = 0; ; ++i) {
        if (readFeature("data/testimg/faceRank/", i) == false) {
            break;
        }
    }

    FrameBatch *fb = fbhelper->getFrameBatch();
    frprocessor->Update(fb);
    for (int i = 0; i < fb->batch_size(); ++i) {
        EXPECT_EQ(0, fb->frames()[i]->objects().size());
    }
    for (int i = 0; i < fb->batch_size(); ++i) {
        FaceRankFrame *frame = (FaceRankFrame *)(fb->frames()[i]);
        frprocessor->Update(frame);
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
