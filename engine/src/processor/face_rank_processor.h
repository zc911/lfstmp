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
    FaceRankProcessor()
            : Processor(),
              detector_("models/shapeface1", "models/avgface1", true, 1,
                        Size(800, 450), 0.7),
              extractor_("models/deployface1", "models/modelface1", true, 1, "",
                         "") {

    }
    virtual ~FaceRankProcessor() {
    }

    virtual void Update(Frame *frame) {
        if (!checkOperation(frame)) {
            LOG(INFO)<< "operation no allowed" << endl;
            return;
        }
        if (!checkStatus(frame)) {
            LOG(INFO) << "check status failed " << endl;
            return;
        }
        LOG(INFO) << "start process frame: " << frame->id() << endl;

        //process frame
        FaceRankFrame *fframe = (FaceRankFrame *)frame;
        fframe->result_ = Rank(fframe->image_, fframe->hotspots_[0], fframe->candidates_);

        frame->set_status(FRAME_STATUS_FINISHED);
        LOG(INFO) << "end process frame: " << frame->id() << endl;
    }

    virtual void Update(FrameBatch *frameBatch) {

    }

    virtual bool checkOperation(Frame *frame) {
        return true;
    }

    virtual bool checkStatus(Frame *frame) {
        return frame->status() == FRAME_STATUS_FINISHED ? false : true;
    }

    void rank(const Mat& image) {


    }

private:
    FaceDetector detector_;
    FaceFeatureExtractor extractor_;

    vector<Score> Rank(const Mat& image, const Rect& hotspot, const vector<FaceFeature>& candidates)
    {
        std::vector<Mat> images;
        images.push_back(image);
        std::vector<FaceFeature> features = extractor_.Extract(images);

        FaceFeature feature = features[0];

            vector<Score> pred;
            for (int i = 0; i < features.size(); i ++) {
                Score p(i, get_cos_similarity(feature.descriptor_, features[i].descriptor_));
                pred.push_back(p);
            }
            return pred;

    }


    float get_cos_similarity(const vector<float> & A, const vector<float> & B) {
        float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
        for (unsigned int i = 0; i<A.size(); ++i) {
            dot += A[i] * B[i];
            denom_a += A[i] * A[i];
            denom_b += B[i] * B[i];
        }
        return abs(dot) / (sqrt(denom_a) * sqrt(denom_b));
    }
};

}

#endif //MATRIX_ENGINE_FACE_RANK_PROCESSOR_H_
