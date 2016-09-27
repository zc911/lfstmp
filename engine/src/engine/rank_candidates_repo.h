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

using namespace std;

namespace dg {

typedef struct {
    unsigned int id;
    string feature;
    string image_uri;
} RankCandidatesItem;

class RankCandidatesRepo {

 public:
    RankCandidatesRepo() { }
    ~RankCandidatesRepo() { }

    void Load() {
        loadFromFile("./repo");
    }
    RankCandidatesItem Get(unsigned int id) {

    }
 private:
    void loadFromFile(const string &folderPath) {
        boost::filesystem::path folder(folderPath);
        if (boost::filesystem::exists(folder) && boost::filesystem::is_directory(folder)) {
            boost::filesystem::directory_iterator itr(folder);
            boost::filesystem::directory_iterator end;
            boost::char_separator<char> sep(" , ");

            typedef boost::tokenizer<boost::char_separator<char>> MyTokenizer;
            string id, feature;

            for (; itr != end; ++itr) {
                LOG(INFO) << "Candidates repo data file: " << *itr << endl;
                string fileName = "./repo/1.txt";
                ifstream file;
                file.open(fileName.c_str());
                string line;
                while (!file.eof()) {
                    getline(file, line);
                    cout << line << endl;
                    MyTokenizer tok(line, sep);
                    MyTokenizer::iterator item = tok.begin();
                    if (item != tok.end()) {
                        id = *item;
                        ++item;
                        if (item != tok.end()) {
                            feature = *item;
                        }
                    }
                    cout << id << " : " << feature << endl;

                }

            }
        } else {
            cout << "Invalid folder path: " << folderPath << endl;
        }
    }
    vector<RankCandidatesItem> candidates_;
};

}

#endif //PROJECT_RANKCANDIDATESREPO_H
