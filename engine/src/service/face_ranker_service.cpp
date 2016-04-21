/*============================================================================
 * File Name   : face_ranker_service.cpp
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#include "face_ranker_service.h"

using namespace dg;

FaceRanker::FaceRanker() :
detector_("models/shapeface1", "models/avgface1"), extractor_("models/deployface1", "models/modelface1")
{
    LOG(INFO) << "initialize face ranker";
}

FaceRanker::~FaceRanker()
{
}

vector<Score> FaceRanker::Rank(const Mat& image, const Rect& hotspot, const vector<FaceFeature>& candidates)
{
    vector<Mat> images;
    images.push_back(image);
    vector<Mat> vFace;
    detector_.Align(images, vFace);

    vector<vector<Score> > prediction;
    extractor_.Classify(vFace, candidates, prediction);

    return prediction[0];
}