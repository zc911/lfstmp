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

#include "model/frame.h"
#include "model/model.h"
#include "processor/processor.h"

namespace dg {

class FaceRankProcessor: public Processor {
 public:
    FaceRankProcessor();
    FaceRankProcessor(float alpha, float beta);
    virtual ~FaceRankProcessor();
 protected:
    virtual bool process(Frame *frame);
    virtual bool process(FrameBatch *frameBatch) {
        return false;
    }
    virtual bool beforeUpdate(FrameBatch *frameBatch);


    virtual bool RecordFeaturePerformance();
    virtual string processorName() {
        return "FaceRankProcessor";
    }
 private:
    float alpha_;
    float beta_;

};

} /* namespace dg */

#endif /* FACE_RANK_PROCESSOR_H_ */
