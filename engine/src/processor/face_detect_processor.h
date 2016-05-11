/*============================================================================
 * File Name   : face_detect_processor.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年3月2日 下午1:53:19
 * Description : 
 * ==========================================================================*/
#ifndef FACE_DETECT_PROCESSOR_H_
#define FACE_DETECT_PROCESSOR_H_

#include "processor/processor.h"
#include "alg/face_detector.h"
#include "model/frame.h"
#include "model/model.h"

namespace dg {

class FaceDetectProcessor : public Processor {
 public:
    FaceDetectProcessor(FaceDetector::FaceDetectorConfig config);
    virtual ~FaceDetectProcessor();

    void Update(Frame *frame);
    void Update(FrameBatch *frameBatch);
    void beforeUpdate(FrameBatch *frameBatch);
    bool checkOperation(Frame *frame) {
        return true;
    }

    bool checkStatus(Frame *frame) {
        return true;
    }

 private:
    FaceDetector *detector_;
    int base_id_;
};

} /* namespace dg */

#endif /* FACE_DETECT_PROCESSOR_H_ */
