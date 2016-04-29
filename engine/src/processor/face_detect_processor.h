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

namespace dg
{

class FaceDetectProcessor : public Processor
{
public:
	FaceDetectProcessor(string model_file, string trained_file,
			const bool use_gpu, const int batch_size, float threshold,
			int width, int height);
	virtual ~FaceDetectProcessor();

	void Update(Frame *frame);
	void Update(FrameBatch *frameBatch)
	{
	}

	bool checkOperation(Frame *frame)
	{
		return true;
	}

	bool checkStatus(Frame *frame)
	{
		return true;
	}

private:
	string model_file_;
	string trained_file_;
	float det_thresh_;
	Size resolution_;
	FaceDetector *detector_;
};

} /* namespace dg */

#endif /* FACE_DETECT_PROCESSOR_H_ */
