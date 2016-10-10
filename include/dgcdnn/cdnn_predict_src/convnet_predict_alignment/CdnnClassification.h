/***************************************************************************
 * CdnnClassification.h
 **************************************************************************/
 
#ifndef __CDNN_CLASSIFICATION_H__
#define __CDNN_CLASSIFICATION_H__

#include <string>
#include <vector>

#include "CdnnSubMean.h"

typedef struct CLASS_MODEL
{
    void* cdnnModel;
    dcnn_predict_params cdnnParams;
}
sClassModel;

bool InitClassModel(const char* modelDir, sClassModel& classModel);
void ReleaseClassModel(sClassModel& model);

bool Class(double& conf, IplImage* srcImage, std::vector<double>& landmarks, sClassModel& classModel, int cropBorder);

#endif 

