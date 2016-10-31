/*============================================================================
 * File Name   : face_quality_processor.h
 * Author      : jiajiachen@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年10月27日 下午2:44:11
 * Description :
 * ==========================================================================*/
#ifndef FACE_POSE_PROCESSOR_H_
#define FACE_POSE_PROCESSOR_H_

#include "model/frame.h"
#include "model/model.h"
#include "processor/processor.h"
#include "dgface/quality/qual_posem.h"


namespace dg {
typedef struct {

} FacePoseConfig;
class FacePoseProcessor: public Processor {
public:
	FacePoseProcessor(
	    const FacePoseConfig &config);
	virtual ~FacePoseProcessor();

protected:
	virtual bool process(Frame *frame) {};
	virtual bool process(FrameBatch *frameBatch);

	virtual bool RecordFeaturePerformance();

	virtual bool beforeUpdate(FrameBatch *frameBatch);

private:
	vector<Object *> to_processed_;
	DGFace::PoseQuality *fp_;
};

} /* namespace dg */

#endif /* FACE_FEATURE_EXTRACT_PROCESSOR_H_ */
