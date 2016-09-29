//
// Created by chenzhen on 9/23/16.
//

#ifndef PROJECT_RANKCANDIDATESREPO_H
#define PROJECT_RANKCANDIDATESREPO_H

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/core/core.hpp>

using namespace std;

namespace dg {

typedef struct {
    unsigned int id = -1;
    string name;
    vector<float> feature;
    cv::Mat image;
    string image_uri;
} RankCandidatesItem;

class CDatabase;

class RankCandidatesRepo {

 public:
    static RankCandidatesRepo &GetInstance() {
        static RankCandidatesRepo repo;
        return repo;
    }
    ~RankCandidatesRepo();

    void Init(const string &repoPath);

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

 private:
    RankCandidatesRepo();
    void loadFromFile(const string &folderPath);
    void initFaceRankDatabase(unsigned int batchSize);

    CDatabase *face_ranker_;
    vector<RankCandidatesItem> candidates_;
    bool is_init_;
};

}

#endif //PROJECT_RANKCANDIDATESREPO_H
