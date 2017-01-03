/*============================================================================
 * File Name   : face_feature_extract_processor.h
 * Author      : tongliu@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 2016年10月21日 下午3:44:11
 * Description :
 * ==========================================================================*/
#ifndef FACE_FEATURE_EXTRACT_PROCESSOR_H_
#define FACE_FEATURE_EXTRACT_PROCESSOR_H_


#include "model/frame.h"
#include "model/model.h"
#include "processor/processor.h"
#include "dgface/recognition.h"

namespace dg {

typedef struct {
    bool is_model_encrypt = true;
    int batch_size = 1;
    string model_file;
    string deploy_file;
    string layer_name;
    vector<float> mean;
    float pixel_scale;
    vector<int> face_size;
    string pre_process;
    bool use_GPU;
    int gpu_id;
    int method;
    string model_dir;
    string model_config;
    bool islog = false;
    bool concurrency = false;

} FaceFeatureExtractorConfig;

class FaceFeatureExtractProcessor: public Processor {
 public:
    enum { CNNRecog = 0, LBPRecog = 1, CDNNRecog = 2, CdnnCaffeRecog = 3, CdnnFuse = 4 };


    FaceFeatureExtractProcessor(
        const FaceFeatureExtractorConfig &config);
    virtual ~FaceFeatureExtractProcessor();

 protected:
    virtual bool process(Frame *frame);
    virtual bool process(FrameBatch *frameBatch);

    virtual bool RecordFeaturePerformance();

    virtual bool beforeUpdate(FrameBatch *frameBatch);
    int toAlignmentImages(vector<Mat> &imgs, vector<DGFace::AlignResult> &align_results);
    int RecognResult2MatrixRecogn(const vector<DGFace::RecogResult> &recog_results, vector<FaceRankFeature> &features);

 private:
    DGFace::Recognition *recognition_ = NULL;
    vector<Object *> to_processed_;
    string pre_process_;
};

} /* namespace dg */

#endif /* FACE_FEATURE_EXTRACT_PROCESSOR_H_ */
