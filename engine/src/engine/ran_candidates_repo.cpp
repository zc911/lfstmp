//
// Created by chenzhen on 9/28/16.
//

#include "rank_candidates_repo.h"
#include <glog/logging.h>
#include "codec/base64.h"
#include "alg/rank/database.h"

namespace dg {

RankCandidatesRepo::RankCandidatesRepo(const string &repoPath) : repo_path_(repoPath) {
    database_ = new CDatabase();
    database_->SetWorkingGPUs(1);

}

RankCandidatesRepo::~RankCandidatesRepo() {
    candidates_.clear();
}

void RankCandidatesRepo::Load() {
    loadFromFile(repo_path_);
}

void RankCandidatesRepo::loadFromFile(const string &folderPath) {
    boost::filesystem::path folder(folderPath);
    if (boost::filesystem::exists(folder) && boost::filesystem::is_directory(folder)) {
        boost::filesystem::directory_iterator itr(folder);
        boost::filesystem::directory_iterator end;
        boost::char_separator<char> sep(" ");

        typedef boost::tokenizer<boost::char_separator<char>> MyTokenizer;
        string idString, feature;

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
            while (!file.eof()) {
                getline(file, line);
                if (line.size() > 0) {
                    MyTokenizer tok(line, sep);
                    MyTokenizer::iterator col = tok.begin();
                    if (col != tok.end()) {
                        idString = *col;
                        ++col;
                        if (col != tok.end()) {
                            feature = *col;
                        }
                    }
                    RankCandidatesItem item;
                    item.image_uri = idString;
                    Base64::Decode(feature, item.feature);
                    candidates_.push_back(item);
                }

            }
        }

        cout << "Candidates repo size: " << candidates_.size() << endl;
        // BASE64 decode bug here, the last item was duplicated
        int featureLen = candidates_[0].feature.size() - 1;
        cout << "Feature length is " << featureLen << endl;

        database_->Initialize(candidates_.size(), featureLen);

        int batchSize = 1024;
        int batchCount = candidates_.size() / batchSize;

        cout << "Batch size: " << batchSize << " and batch count: " << batchCount << endl;

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
            database_->AddItems(batchFeatures, batchIds, batchSize);
        }


        cout << "CDATABASE: " << database_->GetItemCount() << endl;
        int remains = candidates_.size() - id - 1;
        int remains2 = candidates_.size() % batchSize;

        if (remains > 0) {
            cout << "Some candidates remains " << remains << "," << remains2 << endl;
            for (int i = 0; i < remains; ++i) {
                ++id;
                RankCandidatesItem &item = candidates_[id];
                memcpy((char *) batchFeatures + i * featureLen,
                       (char *) item.feature.data(),
                       featureLen * sizeof(float));
                batchIds[i] = id;
            }
            database_->AddItems(batchFeatures, batchIds, remains);
        }

        cout << "CDATABASE: " << database_->GetItemCount() << endl;

        delete[] batchFeatures;
        delete[] batchIds;



    } else {
        cout << "Invalid folder path: " << folderPath << endl;
    }
}

}