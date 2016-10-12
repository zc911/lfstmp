/***************************************************************************
 * 
 * Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file FaceProcess.h
 * @author liujingtuo(com@baidu.com)
 * @date 2014/03/06 22:38:39
 * @brief 
 *  
 **/




#ifndef  __FACEPROCESS_H_
#define  __FACEPROCESS_H_

#include "face_inf.h"
bool DetectInitSingle(const char* pModelPath,SCascadeL& face_cascade);
bool DetectInit(const char* pModelPath,SCascadeL* face_cascade,int threadnum);
int getCdnnRefImage(IplImage* pImg, LandMarkInfo landmarkinfo,RefImageInfo refImageInfo,IplImage** refimage,LandMarkInfo& landmarkinfo_warp);

int face_detect(SCascadeL& face_cascade,IplImage* pImg,vector<DetectedFaceInfo>& faceinfo,double ratio = 1.0,int min_face_size = MIN_FACE_SIZE);
void getMaxIdx(float* array,int len,int& idx,float& value);   












#endif  //__FACEPROCESS_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
