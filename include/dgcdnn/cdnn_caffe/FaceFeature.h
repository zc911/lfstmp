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




#ifndef  __FACEFEATURE_H_
#define  __FACEFEATURE_H_

#include "face_inf.h"
#include "face_para.h"
class FaceFeature
{
    public:
        FaceFeature(){}
        virtual ~FaceFeature(){}
        virtual int ExtractFeature(IplImage* image, FacePara& param, 
LandMarkInfo& landmark, vector<char>& feature){}
        
};
class CaffeFaceFeature:public FaceFeature {
    public:
        CaffeFaceFeature(FacePara& param);
        ~CaffeFaceFeature(){
            delete [] m_CaffeHandler;
            m_CaffeHandler = NULL;
        }
        int ExtractFeature(IplImage* image, FacePara& param, 
LandMarkInfo& landmark, vector<char>& feature);
    protected:
        bool multi_thread_;
        bool multi_process_;
        void * m_CaffeHandler;
        RefImageInfo m_RefImageInfo;
};

#endif  //__FACEFEATURE_H_

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
