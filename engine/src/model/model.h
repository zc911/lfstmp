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

using namespace std;

namespace dg {

typedef enum {
    OBJECT_UNKNOWN = 0,
    OBJECT_CAR = 1,
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
    int id;
    Box box;
    Confidence confidence;
} Detection;

class Object {
 public:
    Object()
            : id_(0),
              type_(OBJECT_UNKNOWN),
              confidence_(0) {
        children_.clear();

    }
    virtual ~Object() {
        // here we only take care of children but not parent
        for (int i = 0; i < children_.size(); ++i) {
            Object * obj = children_[i];
            if (obj) {
                delete obj;
                obj = NULL;
            }
        }
        children_.clear();
    }

    const vector<Object*>& children() const {
        return children_;
    }

    void set_children(const vector<Object*>& children) {
        children_ = children;
    }

    Confidence confidence() const {
        return confidence_;
    }

    void set_confidence(Confidence confidence) {
        confidence_ = confidence;
    }

    const Detection& detection() const {
        return detection_;
    }

    void set_detection(const Detection& detection) {
        detection_ = detection;
    }

    Identification id() const {
        return id_;
    }

    void set_id(Identification id) {
        id_ = id;
    }

    ObjectType type() const {
        return type_;
    }

    void set_type(ObjectType type) {
        type_ = type;
    }

 protected:
    Identification id_;
    Identification class_id_;
    ObjectType type_;
    Confidence confidence_;
    Detection detection_;
    vector<Object *> children_;
};

class Vehicle : public Object {
 public:

    Vehicle()
            : confidence_(0) {
    }

    typedef struct {
        Identification class_id;
        Confidence confidence;
    } Color;

    typedef struct {
        Box box;
        string plate_num;
        int plate_type;
        Confidence confidence;
    } Plate;

 private:
    cv::Mat image_;
    Plate plate_;
    Color color_;
    Confidence confidence_;
};

class People : public Object {
    People() {
    }
};

class Face : public Object {
    Face() {
    }
};

typedef struct {
    Identification id;
    Timestamp timestamp;
    MessageStatus status;
    MetaData *video_meta_data;
    Object *object;
} Message;

}

#endif /* MODEL_H_ */
