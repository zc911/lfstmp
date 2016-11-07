//
// Created by chenzhen on 11/7/16.
//

#ifndef DGFACE_SSD_FACE_DETECTOR_H
#define DGFACE_SSD_FACE_DETECTOR_H

#include "face_detector.h"

namespace dgface {

class SsdFaceDetector: public FaceDetector {
 public:
    SsdFaceDetector();
    ~SsdFaceDetector();
    void Update();
};

}

#endif //DGFACE_SSD_FACE_DETECTOR_H
