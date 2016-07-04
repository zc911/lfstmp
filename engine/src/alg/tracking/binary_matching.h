#ifndef SRC_ALG_BINARY_MATCHING_H_
#define SRC_ALG_BINARY_MATCHING_H_

#define MAX_MATCHING_NUM 101
#define INF 999999999

namespace dg {
class BinaryMatching {

public:
    BinaryMatching() {
    };
    ~BinaryMatching() {
    };
    float match(const int n, const float w[MAX_MATCHING_NUM][MAX_MATCHING_NUM],
                int inv_link[MAX_MATCHING_NUM]);

private:
    int n_;
    int link_[MAX_MATCHING_NUM];
    float lx_[MAX_MATCHING_NUM], ly_[MAX_MATCHING_NUM],
        slack_[MAX_MATCHING_NUM];
    bool visx_[MAX_MATCHING_NUM], visy_[MAX_MATCHING_NUM];
    bool DFS(const int x, const float w[MAX_MATCHING_NUM][MAX_MATCHING_NUM]);

};
}
#endif /* SRC_ALG_BINARY_MATCHING_H_ */
