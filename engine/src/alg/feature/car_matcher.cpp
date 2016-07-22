/*
 * car_matcher.cpp
 *
 *  Created on: Feb 17, 2016
 *      Author: haoquan
 */

#include <glog/logging.h>
#include "car_matcher.h"

namespace dg {
#define FEATURE_NUM_CUDA 256

#if not USE_CUDA
CarMatcher::CarMatcher(unsigned int maxImageCount) {
    feature_num_ = FEATURE_NUM_CUDA;
    max_resize_size_ = 300;
    max_mis_match_ = 50;
    min_remarkableness_ = 0.8;
    max_mapping_offset_ = 50;
    selected_area_weight_ = 50;
    min_score_thr_ = 100;
    profile_time_ = false;
    max_image_count_ = maxImageCount;

}

CarMatcher::~CarMatcher() {
}
#endif

int CarMatcher::ComputeMatchScore(const CarRankFeature &des1,
                                  const CarRankFeature &des2, const Rect &box) {
    if (profile_time_)
        t_profiler_matching_.Reset();
    Rect box1, box2;
    calcNewBox(des1, des2, box, box1, box2);
    int score = 0;
    float max_mapping_offset_rto = (float) max_mapping_offset_/(float) max_resize_size_;
    max_mapping_offset_rto = 2*max_mapping_offset_rto*max_mapping_offset_rto;
    for (int i = 0; i < des1.descriptor_.rows; i++) {
        uint min_dist = 9999;
        uint sec_dist = 9999;
        int min_idx = -1, sec_idx = -1;

        const uchar* query_feat = des1.descriptor_.ptr<uchar>(i);
        for (int j = 0; j < des2.descriptor_.rows; j++) {

        	float pos1_x_rto = (float) des1.position_.at<ushort>(i, 0)/(float) des1.width_;
        	float pos1_y_rto = (float) des1.position_.at<ushort>(i, 1)/(float) des1.height_;
        	float pos2_x_rto = (float) des2.position_.at<ushort>(j, 0)/(float) des2.width_;
        	float pos2_y_rto = (float) des2.position_.at<ushort>(j, 1)/(float) des2.height_;

            if (calcDis2(pos1_x_rto, pos1_y_rto, pos2_x_rto, pos2_y_rto) < max_mapping_offset_rto) {
                const uchar* train_feat = des2.descriptor_.ptr(j);
                uint dist = calcHammingDistance(query_feat, train_feat);
                if (dist < min_dist) {
                    sec_dist = min_dist;
                    sec_idx = min_idx;
                    min_dist = dist;
                    min_idx = j;
                } else if (dist < sec_dist) {
                    sec_dist = dist;
                    sec_idx = j;
                }
            }
    }
        if ((min_dist <= (unsigned int) (min_remarkableness_ * sec_dist))
            && (min_dist <= (unsigned int) max_mis_match_)) {
            if ((inBox(des1.position_.at<ushort>(i, 0),
                       des1.position_.at<ushort>(i, 1), box1))
                && (inBox(des2.position_.at<ushort>(min_idx, 0),
                          des2.position_.at<ushort>(min_idx, 1), box2))) {
                score = score + selected_area_weight_;
            } else
                score++;
        }
    }
    if (profile_time_) {
        t_profiler_str_ = "Matching";
        t_profiler_matching_.Update(t_profiler_str_);
    }
    return score;
}

void CarMatcher::calcNewBox(const CarRankFeature &des1,
                            const CarRankFeature &des2, const Rect &box,
                            Rect &box1, Rect &box2) {
    if (box.x > 0) {
        float resize_rto1 = max(des1.height_, des1.width_);
        resize_rto1 = ((float) max_resize_size_) / resize_rto1;
        box1.x = box.x * resize_rto1;
        box1.y = box.y * resize_rto1;
        box1.width = box.width * resize_rto1;
        box1.height = box.height * resize_rto1;
        float resize_rto2 = max(des2.height_, des2.width_);
        resize_rto2 = ((float) max_resize_size_) / resize_rto2;
        box2.x = box.x * resize_rto2 - max_mapping_offset_;
        box2.y = box.y * resize_rto2 - max_mapping_offset_;
        box2.width = box.width * resize_rto2 + max_mapping_offset_ * 2;
        box2.height = box.height * resize_rto2 + max_mapping_offset_ * 2;
    } else {
        box1.x = 0;
        box1.y = 0;
        box1.width = max_resize_size_;
        box1.height = max_resize_size_;
        box2.x = 0;
        box2.y = 0;
        box2.width = max_resize_size_;
        box2.height = max_resize_size_;
    }
}

vector<int> CarMatcher::computeMatchScoreCpu(
    const CarRankFeature &des, const Rect &in_box,
    const vector<CarRankFeature> &all_des) {
    vector<int> score;
    for (int i = 0; i < all_des.size(); i++) {
        score.push_back(ComputeMatchScore(des, all_des[i], in_box));
    }
    return score;
}

vector<int> CarMatcher::ComputeMatchScore(
    const CarRankFeature &des, const Rect &in_box,
    const vector<CarRankFeature> &all_des) {
#if USE_CUDA
    return computeMatchScoreGpu(des, in_box, all_des);
#else
    return computeMatchScoreCpu(des, in_box, all_des);
#endif
}

}
