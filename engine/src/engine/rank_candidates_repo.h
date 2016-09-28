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
    unsigned int id;
    vector<float> feature;
    cv::Mat image;
    string image_uri;
} RankCandidatesItem;

class CDatabase;

class RankCandidatesRepo {

 public:
    RankCandidatesRepo(const string &repoPath);
    ~RankCandidatesRepo();

    void Load();
    RankCandidatesItem Get(unsigned int id);

    const vector<RankCandidatesItem> &candidates() const {
        return candidates_;
    }


 private:

    void loadFromFile(const string &folderPath);
    string repo_path_;
    CDatabase *database_;
    vector<RankCandidatesItem> candidates_;
};

}

#endif //PROJECT_RANKCANDIDATESREPO_H
