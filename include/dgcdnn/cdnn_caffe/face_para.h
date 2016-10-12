/***************************************************************************
 * 
 * Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
 
 
/**
 * @file face_para.h
 * @author liujingtuo(com@baidu.com)
 * @date 2014/03/06 21:42:38
 * @brief 
 *  
 **/




#ifndef  __FACE_PARA_H_
#define  __FACE_PARA_H_

struct FacePara
{
    int Knn_RetNum;
    int Knn_K;
    float Knn_Max_Thresh;
    float Knn_Alpha;
    int threadNum;
    int opType;
    // the type of face detector used: 1 for cascade; 2 for cdnn; 3 for fcn detector
    int faceDetector_type;
    int align_model_type;  // the type of align model used: 1 for cnn and 2 for cdnn 
    int feature_type; //1 for cdnn 2 for caffe
    int face_type;//1:other id photo;2:idcard photo;3:living;4:other;5:gongan id photo;6:eduphoto;7:handled idcard photo
    char detectPath[256];
    // for fcn detector
    char fcnDetectProtobufPath[256];
    float detectConfThresh; // set the thresh of detector * alignment post processor
    char alignPath[256]; 
    char featurePath[256];
    char warpConfPath[256];
    char faceComparePath[256];
    char blurPath[256];
    int WarpImageBias;
    int gpu;
};
#endif  //__FACE_PARA_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
