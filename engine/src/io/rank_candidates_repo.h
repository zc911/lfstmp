//
// Created by chenzhen on 9/23/16.
//

#ifndef PROJECT_RANKCANDIDATESREPO_H
#define PROJECT_RANKCANDIDATESREPO_H

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <mutex>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/core/core.hpp>
#include "model/frame.h"

using namespace std;

namespace dg {

typedef FaceRankFeature RankCandidatesItem;
//typedef struct {
//    unsigned int id = -1;
//    string name;
//    vector<float> feature;
//    cv::Mat image;
//    string image_uri;
//} RankCandidatesItem;

class CDatabase;

class RankCandidatesRepo {

 public:
    static RankCandidatesRepo &GetInstance() {
        static RankCandidatesRepo repo;
        return repo;
    }
    ~RankCandidatesRepo();

    void Init(const string &repoPath,
              const string &imageRootPath,
              unsigned int capacity,
              unsigned int featureLen,
              bool needSaveToFile,
              unsigned int saveIterval);

    const RankCandidatesItem &Get(unsigned int id) const {
        if (id >= candidates_.size()) {
            return RankCandidatesItem();
        }
        return candidates_[id];

    }

    CDatabase &GetFaceRanker() {
        return *face_ranker_;
    }

    const vector<RankCandidatesItem> &candidates() const {
        return candidates_;
    }

    int RepoSize() {
        return candidates_.size();
    }

    int RepoCapacity() {
        return capacity_;
    };

    int AddFeatures(const FeaturesFrame &frame);

    void FindCandidatesInfoById(const string &idRegExp, vector<RankCandidatesItem> &results);
    void FindCandidatesInfoByName(const string &nameRegExp, vector<RankCandidatesItem> &results);

    void save(const string &repoPath, unsigned int saveIterval);
    void saveToFile(const string &repoPath, unsigned int startIndex, unsigned int endIndex);
 private:
    RankCandidatesRepo();
    void loadFromFile(const string &folderPath);
    void loadFileThread(const string filePath);
    void addDataToFaceRankDatabase(unsigned int batchSize, unsigned int totalSize, unsigned int fromIndex = 0);
    bool checkData(unsigned int index);

    CDatabase *face_ranker_;
    vector<RankCandidatesItem> candidates_;
    std::mutex put_mutex_;
    string repo_path_;
    string image_root_path_;
    unsigned int feature_len_;
    unsigned int gpu_num_;
    unsigned int capacity_;
    bool need_save_to_file_;
    unsigned int save_iterval_;
    unsigned int new_added_index_;
    bool is_init_;
};

}

#endif //PROJECT_RANKCANDIDATESREPO_H
