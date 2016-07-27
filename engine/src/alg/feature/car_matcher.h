/*
 * car_matcher.h
 *
 *  Created on: Feb 17, 2016
 *      Author: haoquan
 */

#ifndef SRC_CAR_MATCHER_H_
#define SRC_CAR_MATCHER_H_

#include <stdio.h>
#include <vector>
#include <math.h>

#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "timing_profiler.h"

#if USE_CUDA
#include <cuda_runtime.h>
#endif

#include "model/rank_feature.h"

using namespace cv;
using namespace std;

namespace dg {

class CarMatcher {
public:
    CarMatcher(unsigned int maxImageNum);
    virtual ~CarMatcher();

//    void ExtractDescriptor(const Mat &img, CarRankFeature &des);

    vector<int> ComputeMatchScore(const CarRankFeature &des, const Rect &in_box,
                                  const vector<CarRankFeature> &all_des);
    vector<int> ComputeMatchScore(const int &query_img_idx, const Rect &in_box,
                                  const vector<CarRankFeature> &all_des) {
        const CarRankFeature &des = all_des[query_img_idx];
        vector<int> score = ComputeMatchScore(des, in_box, all_des);
        score[query_img_idx] = -999;
        return score;
    }

    int ComputeMatchScore(const CarRankFeature &des1,
                          const CarRankFeature &des2, const Rect &box);
    int ComputeMatchScore(const CarRankFeature &des1,
                          const CarRankFeature &des2) {
        LOG(INFO) << "No interest area inputed.";
        return ComputeMatchScore(des1, des2, Rect(-1, -1, -1, -1));
    }
    string getFeatureTimeCost() {
        return (profile_time_) ?
               t_profiler_feature_.getSmoothedTimeProfileString()
                               :
               "TimeProfiling is not opened!";
    }
    string getMatchTimeCost() {
        return (profile_time_) ?
               t_profiler_matching_.getSmoothedTimeProfileString()
                               :
               "TimeProfiling is not opened!";
    }
    int getFeatNum() {
        return feature_num_;
    }

private:
//    ORB orb_;
    HammingLUT lut_;
//    vector<KeyPoint> key_point_;
    Mat descriptor_;
    bool profile_time_;
    string t_profiler_str_;
    TimingProfiler t_profiler_feature_;
    TimingProfiler t_profiler_matching_;
    int feature_num_;
    int max_resize_size_;
    int max_mis_match_;
    float min_remarkableness_;
    int max_mapping_offset_;
    int selected_area_weight_;
    int min_score_thr_;
    int score_[100000];
    unsigned int max_image_num_;

#if USE_CUDA
    cudaStream_t stream_;

    ushort *query_pos_cuda_;
    uchar *query_desc_cuda_;
    ushort *db_pos_cuda_;
    uchar *db_desc_cuda_;
    ushort *db_width_cuda_;
    ushort *db_height_cuda_;
    int *score_cuda_;

    vector<int>
        computeMatchScoreGpu(const CarRankFeature &des, const Rect &in_box, const vector<CarRankFeature> &all_des);
#endif
    vector<int>
        computeMatchScoreCpu(const CarRankFeature &des, const Rect &in_box, const vector<CarRankFeature> &all_des);

    void calcNewBox(const CarRankFeature &des1, const CarRankFeature &des2, const Rect &box, Rect &box1, Rect &box2);

    void calcNewSize(const ushort &ori_height, const ushort &ori_width, Size &new_size) {
        float resize_rto = max(ori_height, ori_width);
        resize_rto = ((float) max_resize_size_) / resize_rto;
        new_size.height = resize_rto * ori_height;
        new_size.width = resize_rto * ori_width;
    }

    unsigned int calcHammingDistance(const unsigned char *a, const unsigned char *b) {
        return lut_(a, b, 32);
    }

    int calcDis2(const ushort &x1, const ushort &y1, const ushort &x2,
                 const ushort &y2) {
        return (((int) x1) - ((int) x2)) * (((int) x1) - ((int) x2))
            + (((int) y1) - ((int) y2)) * (((int) y1) - ((int) y2));
    }

    bool inBox(const ushort &x, const ushort &y, const Rect &box) {
        return (x >= box.x) && (x <= box.x + box.width) && (y >= box.y)
            && (y <= box.y + box.height);
    }
};

}
#endif /* SRC_CAR_MATCHER_H_ */
