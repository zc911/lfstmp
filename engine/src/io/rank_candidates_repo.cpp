//
// Created by chenzhen on 9/28/16.
//

#include "rank_candidates_repo.h"
#include "codec/base64.h"
#include "alg/rank/database.h"
#include "matrix_util/io/uri_reader.h"
#include <opencv2/highgui/highgui.hpp>
#include "log/log_val.h"

namespace dg {

RankCandidatesRepo::RankCandidatesRepo() : is_init_(false) {

}

RankCandidatesRepo::~RankCandidatesRepo() {
    candidates_.clear();
}

void RankCandidatesRepo::Init(const string &repoPath,
                              const string &imageRootPath,
                              unsigned int capacity,
                              unsigned int featureLen) {
    if (is_init_) {
        return;
    }
    is_init_ = true;
    face_ranker_ = new CDatabase();
    repo_path_ = repoPath;
    image_root_path_ = imageRootPath;
    feature_len_ = featureLen;
    capacity_ = capacity;


    VLOG(VLOG_RUNTIME_DEBUG) << "Find gpu num: " << gpu_num_ << " capacity: " << capacity << endl;
    gpu_num_ = face_ranker_->GetGpuCount();
    face_ranker_->Initialize(capacity / gpu_num_ + 1, feature_len_);

    loadFromFile(repo_path_);
    addDataToFaceRankDatabase(1024, candidates_.size());

}

void RankCandidatesRepo::addDataToFaceRankDatabase(unsigned int batchSize,
                                                   unsigned int totalSize,
                                                   unsigned int fromIndex) {

    if (totalSize == 0) {
        LOG(ERROR) << "Add data to face ranker database failed, empty data" << endl;
        return;
    }
    if (totalSize + face_ranker_->GetTotalItems() > capacity_) {
        LOG(ERROR) << "Exceeds the face ranker database capacity, current size: " << face_ranker_->GetTotalItems()
            << " and new features size: " << totalSize << endl;
        return;
    }

    int batchCount = totalSize / batchSize;

    LOG(INFO) << "Batch size: " << batchSize << " and batch count: " << batchCount << endl;

    float *batchFeatures = new float[batchSize * feature_len_];
    int64_t *batchIds = new int64_t[batchSize];

    int64_t id = fromIndex;
    for (int i = 0; i < batchCount; ++i) {
        for (int j = 0; j < batchSize; ++j) {

            RankCandidatesItem &item = candidates_[id];
            if (item.feature_.size() != feature_len_) {
                LOG(ERROR) << "The input feature length not equals to initial feature length " << item.feature_.size()
                    << ":" << feature_len_ << endl;
            }
            memcpy((char *) (batchFeatures + j * feature_len_),
                   (char *) item.feature_.data(),
                   feature_len_ * sizeof(float));
            batchIds[j] = id;
            ++id;
        }
        face_ranker_->AddItems(batchFeatures, batchIds, batchSize);
    }


    LOG(INFO) << "CDATABASE: " << face_ranker_->GetTotalItems() << endl;
    int remains = candidates_.size() - id;
    int remains2 = candidates_.size() % batchSize;

    LOG(INFO) << "Some candidates remains " << remains << "," << remains2 << endl;

    if (remains > 0) {

        for (int i = 0; i < remains; ++i) {
            RankCandidatesItem &item = candidates_[id];
            memcpy((char *) batchFeatures + i * feature_len_,
                   (char *) item.feature_.data(),
                   feature_len_ * sizeof(float));
            batchIds[i] = id;
            ++id;
        }
        face_ranker_->AddItems(batchFeatures, batchIds, remains);
    }

    LOG(ERROR) << "Repo size in total: " << face_ranker_->GetTotalItems() << endl;


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

                    item.name_ = tokens[1];
                    item.image_uri_ = image_root_path_ + "/" + tokens[2];
                    Base64::Decode(tokens[3], item.feature_);
                    if (item.feature_.size() != feature_len_) {
                        LOG(ERROR) << "Feature length load from repo file not equals to initial feature size "
                            << item.feature_.size() << ":" << feature_len_ << endl;
                        LOG(ERROR) << "The invalid feature info: " << item.name_ << " " << item.image_uri_ << endl;
                    }

                    vector<uchar> imageContent;
                    UriReader::Read(item.image_uri_, imageContent, 5);
                    cv::resize(cv::imdecode(cv::Mat(imageContent), 1), item.image_, cv::Size(64, 64));
                    candidates_.push_back(item);
                }

            }
        }

        LOG(INFO) << "Candidates repo size: " << candidates_.size() << endl;

    } else {
        LOG(ERROR) << "Invalid folder path: " << folderPath << endl;
    }
}

int RankCandidatesRepo::AddFeatures(const FeaturesFrame &frame) {

    if (frame.features_.size() == 0) {
        LOG(ERROR) << "The features frame is empty: " << frame.id() << endl;
        return -1;
    }

    if (frame.features_.size() + face_ranker_->GetTotalItems() >= capacity_) {
        LOG(ERROR) << "Exceeds the ranker repo capacity, current size: " << face_ranker_->GetTotalItems()
            << " and new features size: " << frame.features_.size() << endl;
        return -2;
    }

    int fromIndex = candidates_.size();
    for (auto f : frame.features_) {
        candidates_.push_back(f);
    }
    VLOG(VLOG_RUNTIME_DEBUG) << "add features to repo:" << frame.features_.size() << endl;
    VLOG(VLOG_RUNTIME_DEBUG) << "The repo size: " << candidates_.size() << endl;

    VLOG(VLOG_RUNTIME_DEBUG) << "Before Add data to ranker and ranker database size: " << face_ranker_->GetTotalItems() << endl;
    addDataToFaceRankDatabase(1024, frame.features_.size(), fromIndex);
    VLOG(VLOG_RUNTIME_DEBUG) << "After Add data to ranker and ranker database size: " << face_ranker_->GetTotalItems() << endl;
}

}