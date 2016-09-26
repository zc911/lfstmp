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
#include "alg/detector/face_detector.h"
#include "alg/detector/face_dlib_detector.h"
#include "model/frame.h"
#include "model/model.h"

namespace dg {

class FaceDetectProcessor: public Processor {
public:
	FaceDetectProcessor(FaceCaffeDetector::FaceDetectorConfig config);
	FaceDetectProcessor(FaceDlibDetector::FaceDetectorConfig config);

	virtual ~FaceDetectProcessor();

protected:
	virtual bool process(Frame *frame);
	virtual bool process(FrameBatch *frameBatch);

	virtual bool RecordFeaturePerformance();
	virtual bool beforeUpdate(FrameBatch *frameBatch);


private:
	FaceDetector *detector_ = NULL;
	int base_id_;
	vector<Mat> imgs_;
	vector<Object *> objs_;

};

} /* namespace dg */

#endif /* FACE_DETECT_PROCESSOR_H_ */
