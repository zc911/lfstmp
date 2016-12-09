
#ifndef _DGFACESDK_FACE_CONFIG_H_
#define _DGFACESDK_FACE_CONFIG_H_

#include <string>
#include <map>
#ifdef __APPLE__
  #include <json/json.h>
#else
  #include <jsoncpp/json/json.h>
#endif //__APPLE__
#include "anyconversion.h"

namespace DGFace {

const char INI_SEPERATOR = '=';
const string JSON_FILE_POSTFIX = ".json";
const string INI_FILE_POSTFIX = ".txt";

class Config {

 public:
    Config();

    bool Load(string const& configFile);
    bool LoadString(string const& data);
    AnyConversion const& Value(string const& key) const;
    AnyConversion const& Value(const char *keyFormat, int index) const;
    void AddEntry(string key, AnyConversion value);
    bool KeyExist(string const& section, string const& entry);
    void DumpValues();
    void Clear();

 private:
    bool loadJson(string const& configFile);

    bool loadText(string const& configFile);
    string convertTextToJson(string const& configFile);
    bool isControlFlag(string const& str);
    string trim(string const& source, char const* delims = " \t\r\n");
    void parseJsonNode(Json::Value &node, const string prefix);

 private:
    std::map<string, AnyConversion> content_;

};
}
#endif
