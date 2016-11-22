#ifndef CONVERT_UTIL_H_
#define CONVERT_UTIL_H_

#include "../model/model.h"
#include "data_type.h"
#include "../model/rank_feature.h"
#include "rank_feature.h"

namespace dg {

static Detection ConvertDgvehicleDetection(dgvehicle::Detection d) {
    Detection det;
    det.id = d.id;
    det.deleted = d.deleted;
    det.box = d.box;
    det.confidence = d.confidence;
    det.col_ratio = d.col_ratio;
    det.row_ratio = d.row_ratio;
    return det;
}


/*
ushort width_;
ushort height_;
cv::Mat descriptor_;
cv::Mat position_;
*/
static CarRankFeature ConvertDgvehicleCarRankFeature(dgvehicle::CarRankFeature dgFeature) {
    CarRankFeature carRankFeature;
    carRankFeature.width_ = dgFeature.width_;
    carRankFeature.height_ = dgFeature.height_;
    dgFeature.descriptor_.copyTo(carRankFeature.descriptor_);
    dgFeature.position_.copyTo(carRankFeature.position_);
    return carRankFeature;
}

static dgvehicle::CarRankFeature ConvertToDgvehicleCarRankFeature(CarRankFeature feature) {
    dgvehicle::CarRankFeature carRankFeature;
    carRankFeature.width_ = feature.width_;
    carRankFeature.height_ = feature.height_;
    feature.descriptor_.copyTo(carRankFeature.descriptor_);
    feature.position_.copyTo(carRankFeature.position_);
    return carRankFeature;
}

/*
std::vector<float> descriptor_;
*/
static FaceRankFeature ConvertDgvehicleFaceRankFeature(dgvehicle::FaceRankFeature dgFeature) {
    FaceRankFeature faceRankFeature;
    faceRankFeature.descriptor_.clear();
    faceRankFeature.descriptor_.assign(dgFeature.descriptor_.begin(), dgFeature.descriptor_.end());
    
    return faceRankFeature;
}

static dgvehicle::FaceRankFeature ConvertToDgvehicleFaceRankFeature(FaceRankFeature feature) {
    dgvehicle::FaceRankFeature faceRankFeature;
    faceRankFeature.descriptor_.clear();
    faceRankFeature.descriptor_.assign(feature.descriptor_.begin(), feature.descriptor_.end());
    
    return faceRankFeature;
}

/*
int index_;
float score_;
*/
static Score ConvertDgvehicleScore(dgvehicle::Score dgScore) {
    Score score;
    score.index_ = dgScore.index_;
    score.score_ = dgScore.score_;
    
    return score;
}

/*
PedestrianAttribute ConvertDgvehiclePedestrianAttribute(dgvehicle::PedestrianAttribute p) {
    PedestrianAttribute ped;
    ped.index = p.index;
    ped.tagname.assign(p.tagname);
    ped.confidence = p.confidence;
    ped.threshold_lower = p.threshold_lower;
    ped.threshold_upper = p.threshold_upper;
    ped.categoryId = p.categoryId;
    ped.mappingId = p.mappingId;
    return ped;
}  */

}

#endif