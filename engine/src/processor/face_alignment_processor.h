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
    int gpu_id;
    string model_dir;
    float align_threshold = 0;
} FaceAlignmentConfig;

class FaceAlignmentProcessor: public Processor {
 public:

    enum class AlignmentMethod: int { DlibAlign = 0, CdnnAlign = 1, CdnnCaffeAlign = 2 };

    FaceAlignmentProcessor(const FaceAlignmentConfig &faConfig,
                           FaceAlignmentProcessor::AlignmentMethod alignMethod,
                           FaceDetectProcessor::DetectMethod detectMethod);
    ~FaceAlignmentProcessor();
    virtual bool RecordFeaturePerformance();

 private:

    bool process(Frame *frame);
    bool process(FrameBatch *frameBatch);
    bool beforeUpdate(FrameBatch *frameBatch);


 private:

    DGFace::Alignment *alignment_ = NULL;
    vector<Object *> to_processed_;
    AlignmentMethod align_method_;
    FaceDetectProcessor::DetectMethod detect_method_;
    int face_size_length_;
    float align_threshold_;

};

}

#endif //PROJECT_FACE_ALIGNMENT_PROCESSOR_H
