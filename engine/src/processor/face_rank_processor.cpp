/*============================================================================
 * File Name   : face_rank_processor.cpp
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年4月28日 下午4:09:47
 * Description : 
 * ==========================================================================*/
#include "processor/face_rank_processor.h"

namespace dg
{

FaceRankProcessor::FaceRankProcessor()
{
	ranker_ = new FaceRanker();
}

FaceRankProcessor::~FaceRankProcessor()
{
	delete ranker_;
}

void FaceRankProcessor::Update(Frame *frame)
{
	FaceRankFrame *fframe = (FaceRankFrame *)frame;
	fframe->result_ = ranker_->Rank(fframe->datum_, fframe->hotspots_,
			fframe->candidates_);
}

} /* namespace dg */
