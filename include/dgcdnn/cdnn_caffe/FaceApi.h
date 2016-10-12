/***************************************************************************
 * 
 * Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/



/**
 * @file FaceApi.h
 * @author liujingtuo(com@baidu.com)
 * @date 2014/03/07 17:32:38
 * @brief 
 *  
 **/




#ifndef  __FACEAPI_H_
#define  __FACEAPI_H_

#include "FaceDetector.h"
#include "FaceFeature.h"
#include "face_para.h"
#include "FaceAlign.h"
#include "FaceWarp.h"
class FaceHandler
{
  public:
    FaceDetector** m_Detector;
    FaceAlign** m_Align;
    FaceFeature** m_Feature;
    FaceWarp* m_FaceWarp; 
    int ThreadNum;
    FacePara m_param;
    int globalflag;
    FaceHandler()
    {
      ThreadNum = 0;
      globalflag = 0;
      m_Detector = NULL;
      m_Align = NULL;
      m_Feature = NULL;
      m_FaceWarp = NULL;
    }
    ~FaceHandler();
    int Init(const string &conf_path, const string &conf_file);
    int Init(FacePara& param, int opType, int global = 1, int local = 1);
    int FaceProcess(IplImage* image, FacePara& param, int opType, int threadNum,vector<vis_FaceInfo>& faceInfo,int targetFace=-1);
    void setFeatureHandler(FaceHandler* handler)
    {
      handler->m_Feature = this->m_Feature;
    }
};


#endif  //__FACEAPI_H_

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
