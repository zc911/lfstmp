#include <iostream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;
namespace DGFace{

template<typename T>
static inline T from_str(const string &s) {
    T value;
    istringstream ss(s);
    ss >> value;
    return value;
}

template<typename T>
vector<T> split(const string &str, char delimiter) {
    stringstream ss(str);
    vector<T> tokens;
    string token;

    while(getline(ss, token, delimiter)) {
        tokens.push_back(from_str<T>(token));
    }

    return tokens;
}
}
// int main(int argc, const char *argv[]) {
//     string values = "1,2,3,4";
//     vector<int> sep = split<int>(values, ',');
//     for(auto t : sep) {
//         cout << t << endl;
//     }
// }
