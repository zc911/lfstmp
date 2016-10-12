#ifndef __BD_FACEDETECTOR_H__
#define __BD_FACEDETECTOR_H__

#include <vector>
#include <string>

typedef struct tagDetectedFaceInfo
{
    int left;
    int top;
    int width;
    int height;
    float conf;
    float degree;
    int poseView;
    
    tagDetectedFaceInfo()
    {
        left = -1;
        top = -1;
        width = -1;
        height = -1;
        conf = 0.0;
        degree = 0.0;
        poseView = -1;
    }
    
} BD_DetectedFaceInfo;

class BD_FaceDetector{
    
public:
    BD_FaceDetector();
    ~BD_FaceDetector();
    
private:
    static void *mpBoostDetector;
    static void *mpCdnnModel;

public:
    static bool BeValidModel();
    static bool InitModel(std::string modelPath);
    static int DetectFace(void* inputImage, std::vector<BD_DetectedFaceInfo>& faceRects, int confThresh = 500, int minWin = 32, int minStep = 2);
    static void ReleaseModel();
};

#endif

