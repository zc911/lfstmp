/*============================================================================
 * File Name   : face_rank_processor.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_ENGINE_FACE_RANK_PROCESSOR_H_
#define MATRIX_ENGINE_FACE_RANK_PROCESSOR_H_

#include <string>
#include <glog/logging.h>

#include "processor.h"
#include "timing_profiler.h"

#include "model/frame.h"
#include "model/rank_feature.h"
#include "alg/face_feature_extractor.h"
#include "alg/face_detector.h"

using namespace cv;
using namespace std;

namespace dg {

class FaceRankProcessor : public Processor {
 public:
    FaceRankProcessor();
    virtual ~FaceRankProcessor();

    virtual void Update(Frame *frame);

    virtual void Update(FrameBatch *frameBatch);

    virtual void beforeUpdate(FrameBatch *frameBatch);

    bool checkOperation(Frame *frame);

    virtual bool checkStatus(Frame *frame);

 private:
//    FaceDetector *detector_;
    FaceFeatureExtractor *extractor_;

    vector<Score> rank(const Mat& image, const Rect& hotspot,
                       const vector<FaceRankFeature>& candidates);

    float getCosSimilarity(const vector<float> & A, const vector<float> & B);
};

}

#endif //MATRIX_ENGINE_FACE_RANK_PROCESSOR_H_
