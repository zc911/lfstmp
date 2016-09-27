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

#include "alg/feature/face_feature_extractor.h"
#include "alg/feature/face_alignment.h"
#include "model/frame.h"
#include "model/model.h"
#include "processor/processor.h"

namespace dg {

class FaceFeatureExtractProcessor: public Processor {
public:
	FaceFeatureExtractProcessor(
	    const FaceFeatureExtractor::FaceFeatureExtractorConfig &config, const FaceAlignment::FaceAlignmentConfig &faConfig);
	virtual ~FaceFeatureExtractProcessor();

protected:
	virtual bool process(Frame *frame);
	virtual bool process(FrameBatch *frameBatch);

	virtual bool RecordFeaturePerformance();

	virtual bool beforeUpdate(FrameBatch *frameBatch);

private:
	FaceFeatureExtractor *extractor_ = NULL;
	FaceAlignment *alignment_ = NULL;
	vector<Object *> to_processed_;
};

} /* namespace dg */

#endif /* FACE_FEATURE_EXTRACT_PROCESSOR_H_ */
