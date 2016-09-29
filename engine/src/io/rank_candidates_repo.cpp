//
// Created by chenzhen on 9/28/16.
//

#include "rank_candidates_repo.h"
#include <glog/logging.h>
#include "codec/base64.h"
#include "alg/rank/database.h"
#include "matrix_util/io/uri_reader.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace dg {

RankCandidatesRepo::RankCandidatesRepo() : is_init_(false) {

}

RankCandidatesRepo::~RankCandidatesRepo() {
    candidates_.clear();
}

void RankCandidatesRepo::Init(const string &repoPath) {
    if (is_init_) {
        return;
    }
    is_init_ = true;
    face_ranker_ = new CDatabase();
    face_ranker_->SetWorkingGPUs(1);
    loadFromFile(repoPath);
    initFaceRankDatabase(1024);

}

void RankCandidatesRepo::initFaceRankDatabase(unsigned int batchSize) {
    // BASE64 decode bug here, the last item was duplicated
    int featureLen = candidates_[0].feature.size() - 1;
    LOG(INFO) << "Feature length is " << featureLen << endl;

    face_ranker_->Initialize(candidates_.size(), featureLen);

    int batchCount = candidates_.size() / batchSize;

    LOG(INFO) << "Batch size: " << batchSize << " and batch count: " << batchCount << endl;

    float *batchFeatures = new float[batchSize * featureLen];
    int64_t *batchIds = new int64_t[batchSize];

    int64_t id = 0;
    for (int i = 0; i < batchCount; ++i) {
        for (int j = 0; j < batchSize; ++j) {
            id = i * batchSize + j;
            RankCandidatesItem &item = candidates_[id];
            memcpy((char *) (batchFeatures + j * featureLen),
                   (char *) item.feature.data(),
                   featureLen * sizeof(float));
            batchIds[j] = id;
        }
        face_ranker_->AddItems(batchFeatures, batchIds, batchSize);
    }


    LOG(INFO) << "CDATABASE: " << face_ranker_->GetItemCount() << endl;
    int remains = candidates_.size() - id - 1;
    int remains2 = candidates_.size() % batchSize;

    if (remains > 0) {
        LOG(INFO) << "Some candidates remains " << remains << "," << remains2 << endl;
        for (int i = 0; i < remains; ++i) {
            ++id;
            RankCandidatesItem &item = candidates_[id];
            memcpy((char *) batchFeatures + i * featureLen,
                   (char *) item.feature.data(),
                   featureLen * sizeof(float));
            batchIds[i] = id;
        }
        face_ranker_->AddItems(batchFeatures, batchIds, remains);
    }


    delete[] batchFeatures;
    delete[] batchIds;
}

void RankCandidatesRepo::loadFromFile(const string &folderPath) {
    boost::filesystem::path folder(folderPath);
    if (boost::filesystem::exists(folder) && boost::filesystem::is_directory(folder)) {
        boost::filesystem::directory_iterator itr(folder);
        boost::filesystem::directory_iterator end;
        boost::char_separator<char> sep(" ");

        typedef boost::tokenizer<boost::char_separator<char>> MyTokenizer;

        for (; itr != end; ++itr) {

            if (!boost::filesystem::is_regular_file(*itr)) {
                LOG(ERROR) << "Find invalid file in repo directory: " << itr->path().string() << endl;
                continue;
            }
            LOG(INFO) << "Load candidates repo data file: " << *itr << endl;

            string fileName = itr->path().string();
            ifstream file;
            file.open(fileName.c_str());
            string line;
            vector<string> tokens(4);
            while (!file.eof()) {
                getline(file, line);
                if (line.size() > 0) {
                    MyTokenizer tok(line, sep);
                    MyTokenizer::iterator col = tok.begin();
                    int tokenIndex = 0;
                    while (col != tok.end()) {
                        if (tokenIndex >= 4) {
                            LOG(ERROR) << "Line format invalid read from repo file" << endl;
                            break;
                        }
                        tokens[tokenIndex] = *col;
                        col++;
                        tokenIndex++;

                    }
                    RankCandidatesItem item;

                    item.name = tokens[1];
                    item.image_uri = tokens[2];
                    Base64::Decode(tokens[3], item.feature);
                    vector<uchar> imageContent;
                    UriReader::Read(item.image_uri, imageContent, 5);
                    cv::resize(cv::imdecode(cv::Mat(imageContent), 1), item.image, cv::Size(32, 32));
                    candidates_.push_back(item);
                }

            }
        }

        LOG(INFO) << "Candidates repo size: " << candidates_.size() << endl;

    } else {
        cout << "Invalid folder path: " << folderPath << endl;
    }
}

}