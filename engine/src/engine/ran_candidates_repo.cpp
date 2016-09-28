//
// Created by chenzhen on 9/28/16.
//

#include "rank_candidates_repo.h"
#include <glog/logging.h>


namespace dg {

RankCandidatesRepo::RankCandidatesRepo(const string &repoPath) : repo_path_(repoPath) {

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
        string id, feature;

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
                        id = *col;
                        ++col;
                        if (col != tok.end()) {
                            feature = *col;
                        }
                    }
                    RankCandidatesItem item;
                    item.image_uri = id;
                    item.feature = feature;
                    candidates_.push_back(item);
                }

            }
        }
        cout << "Candidates repo size: " << candidates_.size() << endl;

        cout << "sample: " << candidates_[10000].image_uri << " " << candidates_[10000].feature << endl;
    } else {
        cout << "Invalid folder path: " << folderPath << endl;
    }
}

}