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
#include "model/frame.h"
#include "model/model.h"
#include "dgface/detector/det_dlib.h"
#include "dgface/detector/det_rpn.h"
#include "dgface/detector/det_ssd.h"
#include "dgface/detector.h"

namespace dg {
typedef struct {
	bool is_model_encrypt = false;
	int batch_size = 1;
	int gpu_id = 0;
	int img_scale_max = 300;
	int img_scale_min = 240;
	float scale = 1.0f;
	float confidence = 0.7;
	bool use_gpu = true;


	string deploy_file;
	string model_file;
} FaceDetectorConfig;
class FaceDetectProcessor: public Processor {

public:
	enum {DlibMethod = 2, RpnMethod = 0, SsdMethod = 1};


	//FaceDetectProcessor(FaceDetectorConfig config, int method);
	FaceDetectProcessor(FaceDetectorConfig config, int method);

	virtual ~FaceDetectProcessor();

protected:
	virtual bool process(Frame *frame);
	virtual bool process(FrameBatch *frameBatch);

	virtual bool RecordFeaturePerformance();
	virtual bool beforeUpdate(FrameBatch *frameBatch);
	int DetectResult2Detection(const vector<DGFace::DetectResult> &detect_result, vector< vector<Detection> > &detections);
	void enlarge_box(vector<vector<Detection>> boxes, vector<vector<Rect>> &enlarge_boxes) ;
private:
	DGFace::Detector *detector_ = NULL;
	//FaceDetector *detector_ = NULL;
	int base_id_;
	vector<Mat> imgs_;
	vector<Object *> objs_;

	string detect_type_ = "";
};

} /* namespace dg */

#endif /* FACE_DETECT_PROCESSOR_H_ */
