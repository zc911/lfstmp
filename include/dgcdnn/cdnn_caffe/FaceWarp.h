/***************************************************************************
 * 
 * Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file FaceFeature.h
 * @author liujingtuo(com@baidu.com)
 * @date 2014/03/07 11:15:03
 * @brief 
 *  
 **/




#ifndef  __APP_SEARCH_VIS_FACESDK_FACEWARP_H_
#define  __APP_SEARCH_VIS_FACESDK_FACEWARP_H_

#include "face_inf.h"
#include "face_para.h"
class FaceWarp
{
    public:
        FaceWarp(){}
        virtual ~FaceWarp(){}
        virtual int GetWarpImage(IplImage* image,FacePara& param,LandMarkInfo landmark,IplImage** refimage,LandMarkInfo& landmark_warp){}
};
class CdnnFaceWarp:public FaceWarp
{
    protected:
        RefImageInfo m_RefImageInfo;
    public:
        CdnnFaceWarp(char* pFeaturePath);
        int GetWarpImage(IplImage* image, FacePara& param, LandMarkInfo landmark, IplImage** refimage, LandMarkInfo& landmark_warp);
};

#endif  //__FACEFEATURE_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
