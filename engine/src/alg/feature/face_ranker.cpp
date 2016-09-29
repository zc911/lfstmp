///*============================================================================
// * File Name   : face_matcher.cpp
// * Author      : tongliu@deepglint.com
// * Version     : 1.0.0.0
// * Copyright   : Copyright 2016 DeepGlint Inc.
// * Created on  : 2016年4月28日 下午1:04:59
// * Description :
// * ==========================================================================*/
//#include "face_ranker.h"
//
//namespace dg {
//
//FaceRanker::FaceRanker() {
//}
//
//FaceRanker::~FaceRanker() {
//}
//
//
//void FaceRanker::Sort(vector<Score> &scores, int left, int right) {
//    int i = left, j = right;
//    Score tmp;
//    Score pivot = scores[(left + right) / 2];
//    while (i <= j) {
//        while (scores[i].score_ > pivot.score_)
//            i++;
//        while (scores[j].score_ < pivot.score_)
//            j--;
//        if (i <= j) {
//            tmp = scores[i];
//            scores[i] = scores[j];
//            scores[j] = tmp;
//            i++;
//            j--;
//        }
//    };
//    if (left < j)
//        Sort(scores, left, j);
//    if (i < right)
//        Sort(scores, i, right);
//}
//
//vector<Score> FaceRanker::Rank(const FaceRankFeature *datum) {
//
//}
//
//vector<Score> FaceRanker::Rank(const FaceRankFeature &datum,
//                               const vector<Rect> &hotspots,
//                               const vector<FaceRankFeature> &candidates) {
//    vector<Score> scores;
//    for (int i = 0; i < candidates.size(); i++) {
//        Score score(i, CosSimilarity(datum, candidates[i]));
//        scores.push_back(score);
//    }
//
//    Sort(scores, 0, scores.size() - 1);
//    return scores;
//}
//
//float FaceRanker::CosSimilarity(const FaceRankFeature &A,
//                                const FaceRankFeature &B) {
//    float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
//    for (unsigned int i = 0; i < A.descriptor_.size(); ++i) {
//        dot += A.descriptor_[i] * B.descriptor_[i];
//        denom_a += A.descriptor_[i] * A.descriptor_[i];
//        denom_b += B.descriptor_[i] * B.descriptor_[i];
//    }
//    return abs(dot) / (sqrt(denom_a) * sqrt(denom_b));
//}
//
//} /* namespace dg */
