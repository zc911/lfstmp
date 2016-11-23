#ifndef _DGFACESDK_CONFIG_H_
#define _DGFACESDK_CONFIG_H_

#include <map>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <sstream>
#include "common.h"
namespace DGFace{

template<typename T>
static inline T from_str(const std::string &s);
template<typename T>
static inline std::vector<T> split(const std::string &s, char delimiter);

template<typename T>
static inline std::string to_str(T value);

std::string trim_string(const std::string &str);

class Config {
    public:
        virtual ~Config(void);
        static Config *instance(void);

        virtual bool Parse(void) = 0;

        template<typename T>
        T GetConfig(const std::string &key) {
            auto iter = _configs.find(key);
            if (iter == _configs.end())
                throw std::runtime_error("Missing config: " + key + ".");
            return from_str<T>(iter->second);
        }

        template<typename T>
        T GetConfig(const std::string &key, T default_val) {
            auto iter = _configs.find(key);
            if (iter == _configs.end())
                return default_val;
            return from_str<T>(iter->second);
        }

        template<typename T>
        std::vector<T> GetConfigArr(const std::string &key) {
            auto iter = _configs.find(key);
            if (iter == _configs.end())
                throw std::runtime_error("Missing config: " + key + ".");
            return split<T>(iter->second, ',');
        }

        template<typename T>
        std::vector<T> GetConfigArr(const std::string &key, std::vector<T> default_val) {
            auto iter = _configs.find(key);
            if (iter == _configs.end())
                return default_val;
            return split<T>(iter->second, ',');
        }

    protected:
        Config(void);
        void AddConfig(const std::string &key, const std::string &value);
    private:
        std::map<std::string, std::string> _configs;
        static Config* _instance;
};

class FileConfig : public virtual Config {
    public:
        FileConfig(const std::string &filename);
        virtual bool Parse(void);
    private:
        std::string _filename;
};
class StringConfig : public virtual Config {
    public:
        StringConfig(const std::string &content);
        virtual bool Parse(void);
    private:
        std::string _content;
};
}
#include "config.inl.h"

#endif
