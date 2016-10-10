#ifndef _DGFACESDK_CONFIG_H_
#error "Please do not include this file directly, use utils.h instead."
#else
namespace DGFace{

template<typename T>
T from_str(const std::string &s) {
    T value;
    std::istringstream ss(s);
    ss >> value;
    return value;
}

template<typename T>
std::string to_str(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

template<>
bool from_str<bool>(const std::string &s) {
    if (s == "true" || s == "True" || s == "TRUE" || s == "1")
        return true;
    if (s == "false" || s == "False" || s == "FALSE" || s == "0")
        return false;
    throw std::invalid_argument(s);
}

template<typename T>
std::vector<T> split(const std::string &str, char delimiter) {
    std::stringstream ss(str);
    std::vector<T> tokens;
    std::string token;

    while(getline(ss, token, delimiter)) {
        tokens.push_back(from_str<T>(token));
    }

    return tokens;
}
}
#endif

