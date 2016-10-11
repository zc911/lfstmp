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

#include "dgface/alignment.h"
#include "dgface/alignment/align_dlib.h"
#include "dgface/alignment/align_cdnn.h"

#include "model/frame.h"
#include "model/model.h"
#include "processor/processor.h"
#include "dgface/recognition/recog_cnn.h"
#include "dgface/recognition/recog_lbp.h"
#include "dgface/recognition/recog_cdnn.h"

#include "dgface/recognition.h"
#include "dgface/cdnn_score.h"
namespace dg {
typedef struct {
    bool is_model_encrypt = true;
    int batch_size = 1;
    string align_model;
    string align_deploy;
    vector<int> face_size;
    int method;
    string detect_type;

} FaceAlignmentConfig;
typedef struct {
    bool is_model_encrypt = true;
    int batch_size = 1;
    string model_file;
    string deploy_file;
    string layer_name;
    vector <float> mean;
    float pixel_scale;
    vector<int> face_size;
    string pre_process;
    bool use_GPU;
    int gpu_id;
    int method;
    string model_dir;
    string model_config;
} FaceFeatureExtractorConfig;
class FaceFeatureExtractProcessor: public Processor {
public:
    enum {CNNRecog = 0, LBPRecog = 1, CDNNRecog = 2,CdnnCaffeRecog=3};
    enum {DlibAlign = 0, CdnnAlign = 1,CdnnCaffeAlign = 2};

    FaceFeatureExtractProcessor(
        const FaceFeatureExtractorConfig &config, const FaceAlignmentConfig &faConfig);
    virtual ~FaceFeatureExtractProcessor();

protected:
    virtual bool process(Frame *frame);
    virtual bool process(FrameBatch *frameBatch);

    virtual bool RecordFeaturePerformance();

    virtual bool beforeUpdate(FrameBatch *frameBatch);
    int AlignResult2MatrixAlign(const vector<DGFace::AlignResult> &align_result, vector< Mat > &imgs);
    int RecognResult2MatrixRecogn(const vector<DGFace::RecogResult> &recog_results, vector< FaceRankFeature > &features);
    void adjust_box(string detect_type, Rect &box);
private:
    DGFace::Recognition *recognition_ = NULL;
    DGFace::Alignment *alignment_ = NULL;
    vector<Object *> to_processed_;
    string pre_process_;
    int align_method_;
};

} /* namespace dg */

#endif /* FACE_FEATURE_EXTRACT_PROCESSOR_H_ */
