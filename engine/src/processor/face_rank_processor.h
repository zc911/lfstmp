/*============================================================================
 * File Name   : face_rank_processor.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年4月28日 下午4:09:47
 * Description : 
 * ==========================================================================*/
#ifndef FACE_RANK_PROCESSOR_H_
#define FACE_RANK_PROCESSOR_H_

#include "alg/face_ranker.h"
#include "model/frame.h"
#include "model/model.h"
#include "processor/processor.h"

namespace dg {

class FaceRankProcessor : public Processor {
 public:
    FaceRankProcessor();
    virtual ~FaceRankProcessor();

    virtual void Update(Frame *frame);
    virtual void Update(FrameBatch *frameBatch) {

    }

    virtual bool checkStatus(Frame *frame) {
        return true;
    }

 private:
    FaceRanker *ranker_;
};

} /* namespace dg */

#endif /* FACE_RANK_PROCESSOR_H_ */
