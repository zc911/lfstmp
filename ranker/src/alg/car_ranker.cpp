/*============================================================================
 * File Name   : car_ranker.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#include <glog/logging.h>

#include "car_ranker.h"

CarRanker::CarRanker()
{
}

CarRanker::~CarRanker()
{
}

vector<CarScore> CarRanker::Rank(Rect selected_box, Mat img, vector<CarDescriptor> all_des, int loopsize)
{
	CarDescriptor des;
	car_matcher_.extract_descriptor(img, des);
	float resize_rto = 600.0 / (float) max(img.cols, img.rows);
	int offset = (600 - resize_rto * img.cols) / 2;

	selected_box.x = 1.0 * (selected_box.x - offset) / resize_rto;
	selected_box.y = 1.0 * (selected_box.y - offset) / resize_rto;
	selected_box.width = 1.0 * selected_box.width / resize_rto;
	selected_box.height = 1.0 * selected_box.height / resize_rto;
	t_profiler_matching_.Reset();

	vector<int> score = car_matcher_.compute_match_score(des, selected_box, all_des);

	t_profiler_str_ = "TotalMatching";
	t_profiler_matching_.Update(t_profiler_str_);
	
	vector<CarScore> topx(score.size());
	for (int i = 0; i < score.size(); i++)
	{
		CarScore cs;
		cs.index = i;
		cs.score = score[i];
		topx[i] = cs;
	}

	partial_sort(topx.begin(), topx.begin() + loopsize, topx.end());
	topx.resize(loopsize);
	LOG(INFO)<< "Ranking finished, " <<t_profiler_matching_.getSmoothedTimeProfileString();
	return topx;
}

