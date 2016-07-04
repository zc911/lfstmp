#include "binary_matching.h"

namespace dg {
bool BinaryMatching::DFS(const int x,
                         const float w[MAX_MATCHING_NUM][MAX_MATCHING_NUM]) {
    visx_[x] = true;
    for (int y = 1; y <= n_; y++) {
        if (visy_[y])
            continue;
        int t = lx_[x] + ly_[y] - w[x][y];
        if (t == 0) {
            visy_[y] = true;
            if (link_[y] == -1 || DFS(link_[y], w)) {
                link_[y] = x;
                return true;
            }
        } else if (slack_[y] > t)
            slack_[y] = t;
    }
    return false;
}

float BinaryMatching::match(const int n,
                            const float w[MAX_MATCHING_NUM][MAX_MATCHING_NUM],
                            int inv_link[MAX_MATCHING_NUM]) {
    n_ = n;
    for (int i = 1; i <= n_; i++) {
        lx_[i] = -INF;
        ly_[i] = 0;
        link_[i] = -1;
        inv_link[i - 1] = -1;
        for (int j = 1; j <= n_; j++)
            if (w[i][j] > lx_[i])
                lx_[i] = w[i][j];
    }
    for (int x = 1; x <= n_; x++) {
        for (int i = 1; i <= n_; i++)
            slack_[i] = INF;
        while (1) {
            for (int i = 1; i <= n_; i++)
                visx_[i] = visy_[i] = false;
            if (DFS(x, w))
                break;
            int d = INF;
            for (int i = 1; i <= n_; i++)
                if (!visy_[i] && d > slack_[i])
                    d = slack_[i];
            for (int i = 1; i <= n_; i++) {
                if (visx_[i])
                    lx_[i] -= d;
                if (visy_[i])
                    ly_[i] += d;
                else
                    slack_[i] -= d;
            }
        }
    }
    float res = 0;
    for (int i = 1; i <= n_; i++)
        if (link_[i] > -1) {
            res += w[link_[i]][i];
            inv_link[link_[i] - 1] = i - 1;
        }
    return res;
}
}