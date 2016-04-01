/*
 * model.h
 *
 *  Created on: 01/04/2016
 *      Author: chenzhen
 */

#ifndef MODEL_H_
#define MODEL_H_

#include <opencv2/core/core.hpp>
#include "basic.h"

namespace deepglint {

enum ObjectType {
    OBJECT_VEHICLE = 1,
    OBJECT_BICYCLE = 2,
    OBJECT_TRICYCLE = 4,
    OBJECT_PEDESTRIAN = 8,
    OBJECT_PEOPLE = 16,
    OBJECT_FACE = 32,
};

typedef struct {
    Box box;
    Confidence confidence;
} Detection;

typedef struct {
    ObjectType type;
    Detection detection;
    Feature feature;
    cv::Mat pic;
} Object;

typedef struct : public Object {

} Vehicle;

// TODO
typedef struct : public Object {

} People;

// TODO
typedef struct : public Object {

} Face;

}

#endif /* MODEL_H_ */
