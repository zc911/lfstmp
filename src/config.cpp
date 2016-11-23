#include <config.h>
#include <fstream>
#include <sstream>

using namespace std;
namespace DGFace{
Config *Config::_instance = nullptr;

Config *Config::instance(void) {
    if (!_instance)
        throw runtime_error("Config not initialized.");
    return _instance;
}

Config::Config(void) {
    if (!_instance) {
        _instance = this;
    }
}

Config::~Config(void) {
    if (_instance == this) {
        _instance = nullptr;
    }
}

void Config::AddConfig(const string &key, const string &value) {
    _configs[key] = value;
}

FileConfig::FileConfig(const string &name) {
    _filename = name;
}

bool FileConfig::Parse(void) {
    ifstream fin(_filename);
    if (!fin.is_open()) {
        cerr << "Failed to open file " << _filename << endl;
        return false;
    }
    for (string line; getline(fin, line); ) {
        line = trim_string(line);
        if (line.size() < 1 || line[0] == '#')
            continue;

        size_t index = line.find('=');
        if (std::string::npos == index) {
            cerr << "Failed to parse line: " << line << endl;
            continue;
        }
        string key   = trim_string(line.substr(0, index));
        string value = trim_string(line.substr(index + 1));
        AddConfig(key, value);
    }
    return true;
}

StringConfig::StringConfig(const string &content) {
    _content = content;
}

bool StringConfig::Parse(void) {
    istringstream ssin(_content);
    for (string line; getline(ssin, line); ) {
        line = trim_string(line);
        if (line.size() < 1 || line[0] == '#')
            continue;

        size_t index = line.find('=');
        if (std::string::npos == index) {
            cerr << "Failed to parse line: " << line << endl;
            continue;
        }
        string key   = trim_string(line.substr(0, index));
        string value = trim_string(line.substr(index + 1));
        AddConfig(key, value);
    }
    return true;
}

string trim_string(const string &str) {
    if(str.empty())
        return str;
    std::size_t firstScan = str.find_first_not_of(' ');
    std::size_t first     = firstScan == std::string::npos ? str.length() : firstScan;
    std::size_t last      = str.find_last_not_of(' ');
    return str.substr(first, last - first + 1);
}
}
