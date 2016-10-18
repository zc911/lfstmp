/***************************************************************************
 *
 * Copyright (c) 2014 Dengyafeng. All Rights Reserved
 *
 **************************************************************************/

/**
 * @file ExtractMPMetricFeature.cpp
 * @author dengyafeng@gmail.com)
 * @date 2014/10/16 13:57:43
 * @brief
 *
 **/

#ifndef  __EXTRACT_MPMETRIC_FEATURE__
#define  __EXTRACT_MPMETRIC_FEATURE__

#include <string>
#include <vector>

#include "cv.h"
#include "highgui.h"
namespace Cdnn{
namespace MPMetricFeature
{

typedef struct _PARA_MODEL
{
    void* model;
    float* mean;
    std::vector<std::string> paras;
}
sParaModel;

typedef struct _MULTI_PATCH_CDNN_MODEL
{
    std::vector<sParaModel> multiPatchCdnnModel;
    sParaModel metricLearningModel;
}
sMultiPatchCDNNModel;

typedef struct _MODEL_PARA_
{
    std::vector<std::string> paras; // idx1, idx2, inputSize, cropBorder, scale, offset, flip, outputLayerName
    std::string modelPath; // modelPath
    std::string meanPath; // meanPath
}
sModelParas;

typedef struct _MULTIPATCH_PARA_
{
    std::vector<sModelParas> multiCDNNParas;
    sModelParas metricLearningModelParas;
}
sMultiPatchPara;

class ExtractMPMetricFeature
{
public:
    ExtractMPMetricFeature();
    ~ExtractMPMetricFeature();

    int InitModels(const char* configPath, const char* modelDir);
    void ReleaseModels();
    int ExtractFeature(std::vector<float>& metricFeature, const IplImage* srcImage, const std::vector<double>& landmarks, bool multi_thread);
    int ExtractFeatureRaw(std::vector<float>& metricFeature, const IplImage* srcImage, const std::vector<double>& landmarks, bool multi_thread);

    static float ComputeSimilarity(const float* feature0, const float* feature1, int nFeatureLen);
    bool AffineTransform(const IplImage* src, IplImage*& desImage, const std::vector<double>& srcLandmark, std::vector<double>& desLandmark);

    int GetNormFaceWidth();
    int GetNormFaceHeight();

protected:

    bool LoadModels(sMultiPatchCDNNModel& multiPatchModels, std::string configPath, std::string modelDir);
    bool LoadConfig(sMultiPatchPara& multiPatchModelParas, std::string configPath);
    void ReleaseMultiCDNNMetricModels(sMultiPatchCDNNModel& multiPatchModels);
    void ReleaseModel(sMultiPatchCDNNModel& models);
    int ExtractMPFeature(std::vector<float>& metricFeature, sMultiPatchCDNNModel& models, const IplImage* srcImage, const std::vector<double>& landmarks, bool multi_thread);
    void GetMirrorKeyPoints(float& flipPointX, float&flipPointY, int idx1, int idx2, const std::vector<double>& keyPoints);
    static float ComputeL2Dis(const float* feature0, const float* feature1, int nLen);
    static float CompuCosSim(const float* pFeat0, const float* pFeat1, int featDim);
    CvPoint2D32f AffineTransformPoint(CvPoint2D32f srcPoint, const CvMat* warpMat);
    void CreateComputeAffineMat(CvMat*& warpMat, float eye1_x, float eye1_y, float eye2_x, float eye2_y, float mouth_x, float mouth_y);
    void GetRef3Points(float eye1_x, float eye1_y, float eye2_x, float eye2_y, float mouth_x, float mouth_y, float ref_x[3], float ref_y[3], float lab_x[3], float lab_y[3], bool b3Points);
    

    bool SaveMeanImage(char* filePath, float* mean, int width, int height);
    CvRect GetCropRect(int width, int height, CvRect rect, bool isCenter=false, double ratio = 2.0);

public:
    sMultiPatchCDNNModel models;
    bool m_bInitModels;
};

}
}
#endif  //__EXTRACT_MPMETRIC_FEATURE__

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

