/*============================================================================
 * File Name   : face_ranker_service.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_RANKER_FACE_RANKER_SERVICE_H_
#define MATRIX_RANKER_FACE_RANKER_SERVICE_H_

#include <string>

#include "alg/face_extractor.h"
#include "alg/face_detector.h"
#include "ranker_service.h"
#include "timing_profiler.h"

using namespace cv;
using namespace std;

namespace dg 
{

class FaceRanker final : public Ranker<FaceFeature>
{
public:
    FaceRanker();
    virtual ~FaceRanker();

private:
    FaceDetector detector_;
    FaceExtractor extractor_;

    vector<Score> Rank(const Mat& image, const Rect& hotspot, const vector<FaceFeature>& candidates);
};

}

#endif //MATRIX_RANKER_FACE_RANKER_SERVICE_H_