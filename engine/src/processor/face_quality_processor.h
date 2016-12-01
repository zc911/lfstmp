/*============================================================================
 * File Name   : face_quality_processor.h
 * Author      : jiajiachen@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年10月21日 下午6:44:11
 * Description :
 * ==========================================================================*/
#ifndef FACE_QUALITY_PROCESSOR_H_
#define FACE_QUALITY_PROCESSOR_H_

#include "model/frame.h"
#include "model/model.h"
#include "processor/processor.h"
#include "dgface/quality/qual_frontalm.h"


namespace dg {
typedef struct {
    int blurMMethod;
    int frontalMethod;
    float frontalThreshold;
} FaceQualityConfig;
class FaceQualityProcessor: public Processor {
 public:
    enum { FrontalDlib = 0 };
    FaceQualityProcessor(
        const FaceQualityConfig &config);
    virtual ~FaceQualityProcessor();

 protected:
    virtual bool process(Frame *frame) { };
    virtual bool process(FrameBatch *frameBatch);

    virtual bool RecordFeaturePerformance();

    virtual bool beforeUpdate(FrameBatch *frameBatch);

 private:
    vector<Object *> to_processed_;
    DGFace::FrontalMQuality *fq_;
    float frontalThreshold_;
};

} /* namespace dg */

#endif /* FACE_FEATURE_EXTRACT_PROCESSOR_H_ */