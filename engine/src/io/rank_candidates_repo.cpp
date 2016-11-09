//
// Created by chenzhen on 9/28/16.
//

#include "rank_candidates_repo.h"
#include "codec/base64.h"
#include "alg/rank/database.h"
#include "log/log_val.h"
#include <thread>
#include <jsoncpp/json/json.h>
#include <sys/time.h>
#include "debug_util.h"

namespace dg {

RankCandidatesRepo::RankCandidatesRepo() : is_init_(false) {
    need_save_to_file_ = false;
    save_iterval_ = 30 * 60;
    new_added_index_ = 0;
}

RankCandidatesRepo::~RankCandidatesRepo() {
    candidates_.clear();
}

void RankCandidatesRepo::Init(const string &repoPath,
                              const string &imageRootPath,
                              unsigned int capacity,
                              unsigned int featureLen,
                              bool needSaveToFile,
                              unsigned int saveIterval) {
    if (is_init_) {
        return;
    }
    is_init_ = true;
    face_ranker_ = new CDatabase();
    repo_path_ = repoPath;
    image_root_path_ = imageRootPath;
    feature_len_ = featureLen;
    capacity_ = capacity;
    need_save_to_file_ = needSaveToFile;
    save_iterval_ = saveIterval;


    VLOG(VLOG_RUNTIME_DEBUG) << "Find gpu num: " << gpu_num_ << " capacity: " << capacity << endl;
    gpu_num_ = face_ranker_->GetGpuCount();
    face_ranker_->Initialize(capacity / gpu_num_ + 1, feature_len_);
    struct timeval start, finish;
    gettimeofday(&start, NULL);
    loadFromFile(repo_path_);
    gettimeofday(&finish, NULL);
    LOG(INFO) << "Load features from file cost: " << TimeCostInMs(start, finish) << " ms" << endl;
    addDataToFaceRankDatabase(1024, candidates_.size());
    if (need_save_to_file_) {
        std::thread saveThread(&RankCandidatesRepo::save, this, repo_path_, save_iterval_);
        saveThread.detach();
    }

}

void RankCandidatesRepo::save(const string &repoPath, unsigned int saveIterval) {
    while (1) {
        sleep(saveIterval);
        // there is no new added data
        if (new_added_index_ == candidates_.size()) {
            LOG(WARNING) << "There is not new data save to file, wait for another " << saveIterval << " seconds"
                << endl;
            continue;
        }

        saveToFile(repoPath, new_added_index_, candidates_.size());
        //TODO need a lock
        new_added_index_ = candidates_.size();
    }

}

void RankCandidatesRepo::saveToFile(const string &repoPath, unsigned int startIndex, unsigned int endIndex) {

    const time_t t = time(NULL);
    struct tm *current_time = localtime(&t);
    char fileName[256];
    sprintf(fileName,
            "%s/data_%d_%d_%d:%d:%d_from_%d_to_%d.txt",
            repoPath.c_str(),
            current_time->tm_mon + 1,
            current_time->tm_mday,
            current_time->tm_hour,
            current_time->tm_min,
            current_time->tm_sec,
            startIndex,
            endIndex - 1);

    LOG(INFO) << "Save new added data into file " << fileName << endl;

    ofstream file;
    file.open(fileName);
    for (int i = startIndex; i < endIndex; ++i) {
        RankCandidatesItem &item = candidates_[i];
        file << item.id_;
        file << " ";
        file << item.name_ << " ";
        file << item.image_uri_ << " ";
        string feature = Base64::Encode<float>(item.feature_);
        boost::replace_all(feature, "\n", "");
        file << feature << endl;
    }
    file.flush();
    file.close();

}


bool RankCandidatesRepo::checkData(unsigned int index) {

    if (candidates_.size() != face_ranker_->GetTotalItems()) {
        LOG(ERROR) << "Candidates size not equals to ranker database " << candidates_.size() << ":"
            << face_ranker_->GetTotalItems() << endl;
        return false;
    }

    if (index >= candidates_.size()) {
        LOG(ERROR) << "Index exceeds the database size " << index << ":" << candidates_.size() << endl;
        return false;
    }

    RankCandidatesItem &item = candidates_[index];
    if (item.feature_.size() != feature_len_) {
        LOG(ERROR) << "Features length invalid when check data " << item.feature_.size() << ":" << feature_len_ << endl;
        return false;
    }

    vector<float> feature(feature_len_);
    if (face_ranker_->RetrieveItemById(index, feature.data())) {
        for (int i = 0; i < feature_len_; ++i) {
            if (item.feature_[i] - feature[i] >= 0.001) {
                LOG(ERROR) << "Compare feature valid failed " << item.feature_[i] << ":" << feature[i] << endl;
                return false;
            }
        }
    } else {
        LOG(ERROR) << "Retrieve data from ranker database failed" << endl;
        return false;
    }

    return true;

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
            memcpy(((char *) batchFeatures) + (j * feature_len_ * sizeof(float)),
                   (char *) item.feature_.data(),
                   feature_len_ * sizeof(float));
            batchIds[j] = id;
            ++id;
        }
        face_ranker_->AddItems(batchFeatures, batchIds, batchSize);
    }


    LOG(INFO) << "CDATABASE size: " << face_ranker_->GetTotalItems() << endl;
    int remains = candidates_.size() - id;
    int remains2 = candidates_.size() % batchSize;

    LOG(INFO) << "Some candidates remains " << remains << "," << remains2 << endl;

    if (remains > 0) {

        for (int i = 0; i < remains; ++i) {
            RankCandidatesItem &item = candidates_[id];
            memcpy(((char *) batchFeatures) + (i * feature_len_ * sizeof(float)),
                   (char *) item.feature_.data(),
                   feature_len_ * sizeof(float));
            batchIds[i] = id;
            ++id;
        }
        VLOG(VLOG_RUNTIME_DEBUG)
        << "Add items " << remains << " " << batchIds[0] << " " << batchIds[1] << " " << batchFeatures[0]
            << batchFeatures[1]
            << endl;
        face_ranker_->AddItems(batchFeatures, batchIds, remains);
    }


    if (!(checkData(face_ranker_->GetTotalItems() / 3) && checkData(face_ranker_->GetTotalItems() / 255))) {
        LOG(FATAL) << "Check candidates repo failed, ranker must exit!" << endl;
        return;
    }

    LOG(INFO) << "Repo size in total: " << face_ranker_->GetTotalItems() << endl;

    delete[] batchFeatures;
    delete[] batchIds;
}


typedef boost::tokenizer<boost::char_separator<char>> MyTokenizer;

void RankCandidatesRepo::loadFileThread(const string fileName) {
    cout << "load file in thread: " << fileName << endl;
    boost::char_separator<char> sep(" ");
    ifstream file;
    file.open(fileName.c_str());
    string line;
    vector<string> tokens(5);
    while (!file.eof()) {
        getline(file, line);
        if (line.size() > 0) {
            MyTokenizer tok(line, sep);
            MyTokenizer::iterator col = tok.begin();
            int tokenIndex = 0;
            while (col != tok.end()) {
                if (tokenIndex >= 5) {
                    LOG(ERROR) << "Line format invalid read from repo file" << endl;
                    break;
                }
                tokens[tokenIndex] = *col;
                col++;
                tokenIndex++;

            }
            RankCandidatesItem item;
            item.id_ = tokens[0];
            if (item.id_.size() == 0) {
                item.id_ = "0";
            }
            item.name_ = tokens[1];
            item.image_uri_ = image_root_path_ + tokens[2];
            Base64::Decode(tokens[3], item.feature_);
            if (item.feature_.size() != feature_len_) {
                LOG(ERROR) << "Feature length load from repo file not equals to initial feature size "
                    << item.feature_.size() << ":" << feature_len_ << endl;
                LOG(ERROR) << "The invalid feature info: " << item.name_ << " " << item.image_uri_ << endl;
            }

            // parse the person attributes in json format
            string attributes = tokens[4];
            Json::Value root;
            Json::Reader reader;
            if (reader.parse(attributes, root)) {
                Json::Value::Members children = root.getMemberNames();
                for (auto itr = children.begin(); itr != children.end(); ++itr) {
                    string keyName = *itr;
                    Json::Value value = root[keyName];
                    auto attrItr = item.attributes_.find(keyName);
                    if (attrItr != item.attributes_.end()) {
                        if (value.isString())
                            attrItr->second.insert(value.asString());
                        else if (value.isArray()) {
                            for (int i = 0; i < value.size(); ++i) {
                                Json::Value arrayValue = value[i];
                                attrItr->second.insert(arrayValue.asString());
                            }
                        }
                    } else {
                        set<string> attrSet;

                        if (value.isString()) {
                            attrSet.insert(value.asString());
                        } else if (value.isArray()) {
                            for (int i = 0; i < value.size(); ++i) {
                                Json::Value arrayValue = value[i];
                                attrSet.insert(arrayValue.asString());
                            }
                        }
                        item.attributes_.insert(make_pair(keyName, attrSet));
                    }

                }
            }
            VLOG(VLOG_RUNTIME_DEBUG) << "Face info: " << item << endl;

            {
                std::lock_guard<std::mutex> lock(put_mutex_);
                candidates_.push_back(item);
            }

        }

    }
    file.close();
}

void RankCandidatesRepo::loadFromFile(const string &folderPath) {
    boost::filesystem::path folder(folderPath);
    if (boost::filesystem::exists(folder) && boost::filesystem::is_directory(folder)) {
        boost::filesystem::directory_iterator itr(folder);
        boost::filesystem::directory_iterator end;
        vector<std::thread> allThreads;

        for (; itr != end; ++itr) {

            if (!boost::filesystem::is_regular_file(*itr)) {
                LOG(ERROR) << "Find invalid file in repo directory: " << itr->path().string() << endl;
                continue;
            }
            LOG(INFO) << "Load candidates repo data file: " << *itr << endl;

            string fileName = itr->path().string();
            allThreads.emplace_back(std::thread(&RankCandidatesRepo::loadFileThread, this, fileName));

        }

        for(int i = 0; i < allThreads.size(); ++i)
            allThreads[i].join();


        new_added_index_ = candidates_.size();
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
        if (f.feature_.size() != feature_len_) {
            LOG(ERROR) << "Feature len invalid, will not add to database " << f.feature_.size() << ":" << feature_len_
                << endl;
            continue;
        }
        if (f.id_.size() == 0) {
            f.id_ = "0";
        }
        candidates_.push_back(f);
    }

    if (candidates_.size() == 0) {
        LOG(WARNING) << "No features will be added into ranker database " << endl;
        return -1;
    }

    VLOG(VLOG_RUNTIME_DEBUG) << "Add features to repo:" << frame.features_.size() << endl;
    VLOG(VLOG_RUNTIME_DEBUG) << "The repo size: " << candidates_.size() << endl;

    VLOG(VLOG_RUNTIME_DEBUG)
    << "Before Add data to ranker and ranker database size: " << face_ranker_->GetTotalItems() << endl;
    VLOG(VLOG_RUNTIME_DEBUG)
    << "Add features to repo: " << candidates_.size() - fromIndex << endl;
    addDataToFaceRankDatabase(1024, candidates_.size() - fromIndex, fromIndex);
    VLOG(VLOG_RUNTIME_DEBUG)
    << "After Add data to ranker and ranker database size: " << face_ranker_->GetTotalItems() << endl;
}

}