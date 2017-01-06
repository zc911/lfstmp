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

#include "dgface/detector.h"

namespace dg {
typedef struct {

    bool is_model_encrypt = false;
    int batch_size = 1;
    int gpu_id = 0;
    bool use_gpu = true;
    string model_dir;

} FaceDetectorConfig;

class FaceDetectProcessor: public Processor {

 public:
    enum class DetectMethod: unsigned int {
        DlibMethod = 0, RpnMethod = 1, SsdMethod = 2, FcnMethod = 3
    };


    FaceDetectProcessor(FaceDetectorConfig config, DetectMethod method);

    virtual ~FaceDetectProcessor();

 protected:
    virtual bool process(Frame *frame);
    virtual bool process(FrameBatch *frameBatch);

    virtual bool RecordFeaturePerformance();
    virtual bool beforeUpdate(FrameBatch *frameBatch);
    int DetectResult2Detection
        (const vector<DGFace::DetectResult> &detect_result, vector<vector<Detection> > &detections);
 private:
    DGFace::Detector *detector_ = NULL;
    int base_id_;
    vector<Mat> imgs_;
    vector<Object *> objs_;

};

} /* namespace dg */

#endif /* FACE_DETECT_PROCESSOR_H_ */
