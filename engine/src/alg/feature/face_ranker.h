/*============================================================================
 * File Name   : face_matcher.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年4月28日 下午1:04:59
 * Description : 
 * ==========================================================================*/
#ifndef FACE_MATCHER_H_
#define FACE_MATCHER_H_

#include "model/rank_feature.h"

using namespace std;
using namespace cv;

namespace dg {

class FaceRanker {
public:
    FaceRanker();
    virtual ~FaceRanker();

    vector<Score> Rank(const FaceRankFeature *datum);

    vector<Score> Rank(const FaceRankFeature &datum,
                       const vector<Rect> &hotspots,
                       const vector<FaceRankFeature> &candidates);

private:
    float CosSimilarity(const FaceRankFeature &A, const FaceRankFeature &B);
    void Sort(vector<Score> &scores, int left, int right);
};

} /* namespace dg */

#endif /* FACE_MATCHER_H_ */
