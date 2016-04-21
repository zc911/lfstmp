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
#include "alg/face_extractor.h"
#include "alg/face_detector.h"

using namespace cv;
using namespace std;

namespace dg {

class FaceRankProcessor : public Processor {
public:
    FaceRankProcessor()
            :Processor(),
             detector_("models/shapeface1", "models/avgface1"),
             extractor_("models/deployface1", "models/modelface1")
    {

    }
    virtual ~FaceRankProcessor() {}

    virtual void Update(Frame *frame) {
        if (!checkOperation(frame)) {
            LOG(INFO) << "operation no allowed" << endl;
            return;
        }
        if (!checkStatus(frame)) {
            LOG(INFO) << "check status failed " << endl;
            return;
        }
        LOG(INFO) << "start process frame: " << frame->id() << endl;

        //process frame
        FaceRankFrame *fframe = (FaceRankFrame *)frame;
        fframe->result = Rank(fframe->image, fframe->hotspots[0], fframe->candidates);

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

private:
    FaceDetector detector_;
    FaceExtractor extractor_;

    vector<Score> Rank(const Mat& image, const Rect& hotspot, const vector<FaceFeature>& candidates)
    {
        vector<Mat> images;
        images.push_back(image);
        vector<Mat> vFace;
        detector_.Align(images, vFace);

        vector<vector<Score> > prediction;
        extractor_.Classify(vFace, candidates, prediction);

        return prediction[0];
    }

};

}

#endif //MATRIX_ENGINE_FACE_RANK_PROCESSOR_H_