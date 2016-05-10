/*
 * processor_helper.h
 *
 *  Created on: Apr 29, 2016
 *      Author: jiajiachen
 */

#ifndef SRC_PROCESSOR_PROCESSOR_HELPER_H_
#define SRC_PROCESSOR_PROCESSOR_HELPER_H_
#include "processor.h"
namespace dg {
static int vote(vector<vector<Prediction> > &src,
                vector<vector<Prediction> > &dst, int factor) {
    if (src.size() > dst.size()) {
        for (int i = 0; i < src.size(); i++) {
            vector<Prediction> tmpSrc = src[i];
            vector<Prediction> tmpDst;
            for (int j = 0; j < tmpSrc.size(); j++) {
                tmpDst.push_back(
                        pair<int, float>(tmpSrc[j].first,
                                         tmpSrc[j].second / factor));
            }
            dst.push_back(tmpDst);
        }
        return 1;
    }
    for (int i = 0; i < src.size(); i++) {

        vector<Prediction> tmpSrc = src[i];
        vector<Prediction> tmpDst = dst[i];
        if (tmpSrc.size() != tmpDst.size()) {
            return -1;
        }
        for (int j = 0; j < tmpSrc.size(); j++) {
            tmpDst[j].second += tmpSrc[j].second / factor;
        }
        dst[i] = tmpDst;
    }
    return 1;
}
}

#endif /* SRC_PROCESSOR_PROCESSOR_HELPER_H_ */
