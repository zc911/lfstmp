/***************************************************************************
 * 
 * Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file FaceDetector.h
 * @author liujingtuo(com@baidu.com)
 * @date 2014/03/06 18:00:18
 * @brief 
 *  
 **/


#ifndef  __FACEDETECTOR_H_
#define  __FACEDETECTOR_H_

#include "face_inf.h"
#include "face_para.h"
// #include "XF_DataStruct.h"
// #include "BD_FaceDetector.h"
#include "fcn_detector.h"

class FaceDetector
{
    public:
        FaceDetector(){}
        virtual ~FaceDetector(){};
        virtual int Detect(IplImage* image,FacePara& param,vector<DetectedFaceInfo>& detectInfo){}
};

class SCascadeDetector:public FaceDetector
{
    protected:
        SCascadeL m_DetectHandler;
    public:
        SCascadeDetector(char* pFilename);
        ~SCascadeDetector(); 
        int Detect(IplImage* image,FacePara& param,vector<DetectedFaceInfo>& detectInfo);
};
//added by dengyafeng @ 2015-10-25
class FCNFaceDetector : public FaceDetector
{
    public:
        FCNFaceDetector(string protobufPath, string modelPath)
        {
            _detector = new FCNDetector(protobufPath, modelPath);
        };
        ~FCNFaceDetector()
        {
            if (_detector)
            {
                delete _detector;
            }
        }
        int Detect(IplImage* image, FacePara& param, vector<DetectedFaceInfo>& detectInfo);
    protected:
        FCNDetector* _detector;
};

#endif  //__FACEDETECTOR_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
