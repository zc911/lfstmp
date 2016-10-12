/***************************************************************************
 * 
 * Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
/**
 * @file FCNDetector.h
 * @author dengyafeng(com@baidu.com)
 * @date 2015/10/21 18:00:18
 * @brief 
 *  
 **/

#ifndef  __FCN_DETECTOR_H_
#define  __FCN_DETECTOR_H_

#include <string>
#include <vector>

typedef struct FCNDetectedFace
{
    float left;
    float top;
    float width;
    float height;
    float conf;
    int degree;
    int pose;
}
FCNDetectedFace;

class FCNDetector
{
    public:
        FCNDetector(std::string prototxtPath, std::string modelPath);
        ~FCNDetector(); 
        void init(std::string& model_def, std::string& weight_file);
        int detect(void*image, std::vector<FCNDetectedFace>& detectFaces, float minFaceSize, float maxFaceSize);
    protected:
        void* _detector_handle;
};

#endif  //__FCN_DETECTOR_H_

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

