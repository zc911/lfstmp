/*
 * model.h
 *
 *  Created on: 01/04/2016
 *      Author: chenzhen
 */

#ifndef MODEL_H_
#define MODEL_H_

#include <vector>
#include <opencv2/core/core.hpp>
#include "basic.h"

namespace deepglint {

using namespace std;

typedef enum {
    OBJECT_VEHICLE = 1,
    OBJECT_BICYCLE = 2,
    OBJECT_TRICYCLE = 4,
    OBJECT_PEDESTRIAN = 8,
    OBJECT_MARKER_0 = 16,
    OBJECT_MARKER_1 = 16,
    OBJECT_MARKER_2 = 16,
    OBJECT_MARKER_3 = 16,
    OBJECT_MARKER_4 = 16,
    OBJECT_PEOPLE = 32,
    OBJECT_FACE = 64,
} ObjectType;

typedef struct {
    Box box;
    Confidence confidence;
} Detection;

typedef struct Object {
    virtual ~Object() {
        // here we only take care of children but not parent
        for (int i = 0; i < children.size(); ++i) {
            Object * obj = children[i];
            if (obj) {
                delete obj;
                obj = NULL;
            }
        }
        children.clear();
    }
    Identification id;
    ObjectType type;
    Confidence confidence;
    Detection detection;
    // Feature feature;
    // cv::Mat pic;
    vector<Object *> children;
    Object *parent;
} Object;

typedef struct Vehicle : public Object {
    Vehicle() {
        type = OBJECT_VEHICLE;
    }
} Vehicle;

typedef struct People : public Object {
    People() {
        type = OBJECT_PEOPLE;
    }
} People;

typedef struct Face : public Object {
    Face() {
        type = OBJECT_FACE;
    }
} Face;

}

#endif /* MODEL_H_ */
