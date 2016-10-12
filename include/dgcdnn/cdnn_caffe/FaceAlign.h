/*
 * 
 * Copyright (c) 2014 Baidu.com, Inc. All Rights ReservedAAAAA
 * 
 **************************************************************************/
 
 
 
/**
 * @file FaceAlign.h
 * @author liujingtuo(com@baidu.com)
 * @date 2014/03/07 10:53:57
 * @brief 
 *  
 **/




#ifndef  __FACEALIGN_H_
#define  __FACEALIGN_H_

#include "face_inf.h"
#include "face_para.h"
#include "cdnn_position.h"
#include "linear.h"
#include "post_filter.h"
#include "landmark_detector.h"
class FaceAlign
{
    public:
        FaceAlign(){}
        virtual ~FaceAlign(){}
        virtual int Align(IplImage* image,FacePara& param,DetectedFaceInfo& detect_box,LandMarkInfo& landmark){}
};

//Added by ZhuFuguo@ 2015-12-4 for face alignment with occlusion label
class FaceAlignWithConf:public FaceAlign
{
public:
    FaceAlignWithConf()
    {
        caffe_model = NULL;
    }
    FaceAlignWithConf(char* modelPath);
    ~FaceAlignWithConf()
    {
        delete caffe_model;
        caffe_model = NULL;
    };
    int Align(IplImage* image,FacePara& param,DetectedFaceInfo& detect_box,LandMarkInfo& landmark);
    void init_model(char* modelDir);
private: 
    void predict(IplImage* img,DetectedFaceInfo& detect_box,LandMarkInfo& landmark);
private:
    LandmarkDetector* caffe_model;
};


#endif  //__FACEALIGN_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

