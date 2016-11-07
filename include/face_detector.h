//
// Created by chenzhen on 11/7/16.
//

#ifndef DGFACE_FACE_DETECTOR_H
#define DGFACE_FACE_DETECTOR_H


namespace dgface {

/*
 * This class is the base interface of detecor
 */
class FaceDetector {
 public:
    FaceDetector();
    virtual ~FaceDetector() = 0;
    virtual void Update() = 0;
};

}

#endif //DGFACE_FACE_DETECTOR_H
