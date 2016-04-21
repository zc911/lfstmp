/*
 * car_matcher.h
 *
 *  Created on: Feb 17, 2016
 *      Author: haoquan
 */

#ifndef SRC_CAR_MATCHER_H_
#define SRC_CAR_MATCHER_H_

#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <stdio.h>
#if USE_CUDA
#include <cuda_runtime.h>
#endif

#include "timing_profiler.h"
#include "model/rank_feature.h"

using namespace cv;
using namespace std;
using namespace dg;

#if USE_CUDA
#define CUDA_CALL(value) {	\
cudaError_t _m_cudaStat = value;	\
if (_m_cudaStat != cudaSuccess) {	\
	fprintf(stderr, "Error %s at line %d in file %s\n",	\
			cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);	\
	exit(1);	\
}}
#endif

class CarMatcher
{
public:
	CarMatcher();
	virtual ~CarMatcher();
	
	void extract_descriptor(const Mat &img, CarFeature &des);

	vector<int> compute_match_score(const CarFeature &des, const Rect &in_box, const vector<CarFeature> &all_des);
	vector<int> compute_match_score(const int &query_img_idx, const Rect &in_box, const vector<CarFeature> &all_des)
	{
		const CarFeature &des = all_des[query_img_idx];
		vector<int> score = compute_match_score(des, in_box, all_des);
		score[query_img_idx] = -999;
		return score;
	}

	int compute_match_score(const CarFeature &des1, const CarFeature &des2, const Rect &box);
	int compute_match_score(const CarFeature &des1, const CarFeature &des2)
	{
		LOG(INFO)<<"No interest area inputed.";
		return compute_match_score(des1, des2, Rect(-1, -1, -1, -1));
	}
	string get_feature_time_cost()
	{
		return (profile_time_) ?
			t_profiler_feature_.getSmoothedTimeProfileString()
			:
			"TimeProfiling is not opened!";
	}
	string get_match_time_cost()
	{
		return (profile_time_) ?
			t_profiler_matching_.getSmoothedTimeProfileString()
			:
			"TimeProfiling is not opened!";
	}
	int get_feat_num()
	{
		return feature_num_;
	}

private:
	ORB orb_;
	HammingLUT lut_;
	vector<KeyPoint> key_point_;
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
	int score_[100000];

	void calc_new_box(const CarFeature &des1, const CarFeature &des2, const Rect &box, Rect &box1, Rect &box2);

	void calc_new_size(const ushort &ori_height, const ushort &ori_width, Size &new_size)
	{
		float resize_rto = max(ori_height, ori_width);
		resize_rto = ((float) max_resize_size_) / resize_rto;
		new_size.height = resize_rto * ori_height;
		new_size.width = resize_rto * ori_width;
	}

	unsigned int calc_hamming_distance(const unsigned char* a, const unsigned char* b) 
	{
		return lut_(a, b, 32);
	}

	int calc_dis2(const ushort &x1, const ushort &y1, const ushort &x2,
			const ushort &y2)
	{
		return (((int) x1) - ((int) x2)) * (((int) x1) - ((int) x2))
			 + (((int) y1) - ((int) y2)) * (((int) y1) - ((int) y2));
	}

	bool is_in_box(const ushort &x, const ushort &y, const Rect &box)
	{
		return (x >= box.x) && (x <= box.x + box.width) && (y >= box.y)
			&& (y <= box.y + box.height);
	}

#if USE_CUDA
	cudaStream_t stream_;

	ushort *query_pos_cuda;
	uchar *query_desc_cuda;
	ushort *db_pos_cuda;
	uchar *db_desc_cuda;
	ushort *db_width_cuda;
	ushort *db_height_cuda;
	int *score_cuda;

	vector<int> compute_match_score_gpu(const CarFeature &des, const Rect &in_box, const vector<CarFeature> &all_des);
#endif
	vector<int> compute_match_score_cpu(const CarFeature &des, const Rect &in_box, const vector<CarFeature> &all_des);
};

#endif /* SRC_CAR_MATCHER_H_ */
