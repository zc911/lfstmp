/**
 *     File Name:  file_reader.h
 *    Created on:  07/25/2016
 *        Author:  Xiaodong Sun
 */

#ifndef TEST_FILE_READER_H_
#define TEST_FILE_READER_H_

#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <vector>

using namespace std;

class FileReader {
public:
    FileReader(string fileName) {
        file.open(fileName);
    }

    ~FileReader() {
        file.close();
    }

    void read(string pattern) {
        while (!file.eof()) {
            vector<string> V;
            string str;
            getline(file, str);
            if (str.find(pattern) == string::npos) {
                continue;
            }
            split(str, pattern, V);
            mp[str.substr(0, str.find(pattern))] = V;
        }
    }

    bool is_open() {
        return file.is_open();
    }

    vector<string> getValue(string key) {
        return mp[key];
    }

    int getIntValue(string key, int index) {
        int ans = 0;
        string tmp = mp[key][index];
        for (int i = 0; i < tmp.size(); ++i)
            ans = ans * 10 + tmp[i] - '0';
        return ans;
    }

    void show() {
        map<string, vector<string> >::iterator itor = mp.begin();
        for (; itor != mp.end(); itor++) {
            cout << (*itor).first << " : ";
            for (int i = 0; i < (*itor).second.size(); ++i) {
                cout << (*itor).second[i] << ' ';
            }
            cout << endl;
        }
    }

private:
    string fileName;
    ifstream file;
    map<string, vector<string> > mp;

    void split(const string & str, const string & pattern, vector<string> & V) {
        int pos1 = str.find(pattern);
        int pos2 = pos1;
        while (string::npos != pos2) {
            if (pos2 != pos1) {
                V.push_back(str.substr(pos1, pos2 - pos1));
            }
            pos1 = pos2 + 1;
            pos2 = str.find(pattern, pos1);
        }
        V.push_back(str.substr(pos1, str.size() - pos1));
    }
};

#endif
