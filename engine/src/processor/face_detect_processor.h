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
//#include "alg/detector/face_detector.h"
#include "model/frame.h"
#include "model/model.h"
#include "algorithm_factory.h"

namespace dg {

class FaceDetectProcessor: public Processor {
public:
	FaceDetectProcessor();
	virtual ~FaceDetectProcessor();

protected:
	virtual bool process(Frame *frame);
	virtual bool process(FrameBatch *frameBatch);

	virtual bool RecordFeaturePerformance();
	virtual bool beforeUpdate(FrameBatch *frameBatch);


private:
	dgvehicle::AlgorithmProcessor *detector_ = NULL;
	int base_id_;
};

} /* namespace dg */

#endif /* FACE_DETECT_PROCESSOR_H_ */
