/***************************************************************************
 * CdnnPosition.h
 **************************************************************************/
 
#ifndef __CDNN_POSITION_H__
#define __CDNN_POSITION_H__

#include <string>
#include <vector>

#include "CdnnSubMean.h"

typedef struct KEYPOINTS_MODEL
{
    void* rectModel;
    dcnn_predict_params rectCdnnParams;
    void* fineRectModel;
    dcnn_predict_params fineRectCdnnParams;
    void* ptsModel;
    dcnn_predict_params ptsCdnnParams;
}
sKeyPointsModel;

bool InitFacialLandmarkModel(const char* modelDir, sKeyPointsModel& keyPointsModel);
void ReleaseKeyPointsModel(sKeyPointsModel& model);

bool PredictLandmarks(std::vector<double>& ptsArray, IplImage* srcImage, sKeyPointsModel& keyPointsModel, int cropBorder);
bool PredictFacialLandmarks(CvRect& rect, std::vector<double>& ptsArray, IplImage* srcImage, sKeyPointsModel& keyPointsModel, int cropBorder);

#endif 

