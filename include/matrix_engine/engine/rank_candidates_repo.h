//
// Created by chenzhen on 9/23/16.
//

#ifndef PROJECT_RANKCANDIDATESREPO_H
#define PROJECT_RANKCANDIDATESREPO_H

#include <vector>
#include <boost/filesystem.hpp>

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
            for (; itr != end; ++itr) {
                cout << "File: " << *itr << endl;
            }
        } else {
            cout << "Invalid folder path: " << folderPath << endl;
        }
    }
    vector<RankCandidatesItem> candidates_;
};

}

#endif //PROJECT_RANKCANDIDATESREPO_H
