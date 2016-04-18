/*============================================================================
 * File Name   : car_ranker.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/
#ifndef MATRIX_RANKER_ALG_CAR_RANKER_H_
#define MATRIX_RANKER_ALG_CAR_RANKER_H_

#include "car_matcher.h"

struct CarScore
{
	int index;
	float score;

    // sortable
    bool operator<(const CarScore& right) const 
    { 
        return  score > right.score && index < right.index; 
    }
};

class CarRanker
{
public:
	CarRanker();
	virtual ~CarRanker();

	vector<CarScore> Rank(Rect selected_box, Mat img, vector<CarDescriptor> all_des, int limit);

private:
	string t_profiler_str_;
	TimingProfiler t_profiler_matching_;
	CarMatcher car_matcher_;
};

#endif // MATRIX_RANKER_ALG_CAR_RANKER_H_
