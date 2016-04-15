// Parse and modify *.ini files
#ifndef CONFIGFILE_H
#define CONFIGFILE_H

#include <string>
#include <map>
#include "jsoncpp/json/json.h"
#include "anyconversion.h"

namespace dg {

const char INI_SEPERATOR = '=';
const string JSON_FILE_POSTFIX = ".json";
const string INI_FILE_POSTFIX = ".txt";

class Config {

 public:
    static Config* GetInstance() {
        if (!instance_)
            instance_ = new Config;
        return instance_;
    }

    bool Load(string const& configFile);
    AnyConversion const& Value(string const& key) const;
    AnyConversion const& Value(const char *keyFormat, int index) const;
    void AddEntry(string key, AnyConversion value);
    bool KeyExist(string const& section, string const& entry);
    void DumpValues();

 private:
    Config();

    bool loadText(string const& configFile);
    bool loadJson(string const& configFile);
    string convertTextToJson(string const& configFile);
    bool isControlFlag(string const& str);
    string trim(string const& source, char const* delims = " \t\r\n");
    void parseJsonNode(Json::Value &node, const string prefix);

 private:
    std::map<string, AnyConversion> content_;
    static Config* instance_;

};
}
#endif
