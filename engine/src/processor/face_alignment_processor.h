//
// Created by chenzhen on 12/2/16.
//

#ifndef PROJECT_FACE_ALIGNMENT_PROCESSOR_H
#define PROJECT_FACE_ALIGNMENT_PROCESSOR_H

#include "model/frame.h"
#include "model/model.h"
#include "processor/processor.h"
#include "dgface/alignment.h"
#include "processor/face_detect_processor.h"

namespace dg {

typedef struct {
    bool is_model_encrypt = true;
    int batch_size = 1;
    string align_model;
    string align_deploy;
    string align_cfg;
    string align_path;
    vector<int> face_size;
    int method;
    string detect_type;
    int gpu_id;
    float threshold = 0;
    string align_model_path;
    string model_dir;
} FaceAlignmentConfig;

class FaceAlignmentProcessor: public Processor {
 public:

    enum { DlibAlign = 0, CdnnAlign = 1, CdnnCaffeAlign = 2 };

    FaceAlignmentProcessor(const FaceAlignmentConfig &faConfig, FaceDetectProcessor::FaceDetectMethod detectMethod);
    ~FaceAlignmentProcessor();
    virtual bool RecordFeaturePerformance();

 private:

    bool process(Frame *frame);
    bool process(FrameBatch *frameBatch);
    bool beforeUpdate(FrameBatch *frameBatch);

    static void adjust_box(string detect_type, Rect &adjust_box) {
        if (detect_type == "rpn") {

            ///////////////////////////////////////////////////
            // adjust bounding box
            const float h_rate = 0.30;
            const float w_rate = 0.15;

            float a_dist = adjust_box.height * h_rate;

            adjust_box.y += a_dist;
            adjust_box.height -= a_dist;

            a_dist = adjust_box.width * w_rate;
            adjust_box.x += a_dist;
            adjust_box.width -= a_dist * 2;
        } else if (detect_type == "ssd") {
            float a_dist = adjust_box.height * 0.40;

            adjust_box.y += a_dist;
            adjust_box.height -= a_dist;

            a_dist = adjust_box.width * 0.1;
            adjust_box.x += a_dist;
            adjust_box.width -= a_dist * 2;
        }
    }


 private:

    DGFace::Alignment *alignment_ = NULL;
    vector<Object *> to_processed_;
    int align_method_;
    int face_size_length_;
    float align_threshold_;
    FaceDetectProcessor::FaceDetectMethod detect_method_;
};

}

#endif //PROJECT_FACE_ALIGNMENT_PROCESSOR_H
