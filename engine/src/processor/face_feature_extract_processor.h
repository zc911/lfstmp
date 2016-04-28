/*============================================================================
 * File Name   : face_feature_extract_processor.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年4月21日 下午3:44:11
 * Description : 
 * ==========================================================================*/
#ifndef FACE_FEATURE_EXTRACT_PROCESSOR_H_
#define FACE_FEATURE_EXTRACT_PROCESSOR_H_

#include "alg/face_feature_extractor.h"
#include "model/frame.h"
#include "model/model.h"
#include "processor.h"

namespace dg {

class FaceFeatureExtractProcessor : public Processor {
 public:
    FaceFeatureExtractProcessor(const string& model_file,
                                const string& trained_file, const bool use_gpu,
                                const int batch_size, const string &align_model,
                                const string &avg_face);
    virtual ~FaceFeatureExtractProcessor();

    void Update(Frame *frame);
    virtual void Update(FrameBatch *frameBatch) {

    }

    virtual bool checkOperation(Frame *frame) {
        return true;
    }
    ;
    virtual bool checkStatus(Frame *frame) {
        return true;
    }
    ;

 private:
    FaceFeatureExtractor *extractor_;
};

} /* namespace dg */

#endif /* FACE_FEATURE_EXTRACT_PROCESSOR_H_ */
