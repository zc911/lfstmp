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

typedef struct Detection {
    int id;
    bool deleted;
    Box box;
    Confidence confidence;
    Detection& operator =(const Detection &detection) {
        if (this == &detection) {
            return *this;
        }
        id = detection.id;
        box = detection.box;
        confidence = detection.confidence;
        return *this;
    }
} Detection;

class Object {
 public:
    Object(ObjectType type)
            : id_(0),
              type_(type),
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
    ObjectType type_;
    Confidence confidence_;
    Detection detection_;
    vector<Object *> children_;
};

class Vehicle : public Object {
 public:

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

    Vehicle(ObjectType type)
            : Object(type),
              confidence_(0),
              class_id_(-1) {
    }

    const Color& color() const {
        return color_;
    }

    void set_color(const Color& color) {
        color_ = color;
    }

    Confidence confidence1() const {
        return confidence_;
    }

    void set_confidence(Confidence confidence) {
        confidence_ = confidence;
    }

    const cv::Mat& image() const {
        return image_;
    }

    void set_image(const cv::Mat& image) {
        image_ = image;
    }

    const Plate& plate() const {
        return plate_;
    }

    void set_plate(const Plate& plate) {
        plate_ = plate;
    }

    Identification class_Id() const {
        return class_id_;
    }

    void set_class_id(Identification classId) {
        class_id_ = classId;
    }

 private:

    cv::Mat image_;
    Identification class_id_;
    Plate plate_;
    Color color_;
    Confidence confidence_;
};

class Face : public Object {

 public:
    Face()
            : Object() {

    }
    Face(Identification id, int x, int y, int width, int height,
         Confidence confidence)
            : Object(),
              type_(OBJECT_FACE) {
        id_ = id;
        confidence_ = confidence;
        detection_.box = Box(x, y, width, height);
    }

    FaceFeature feature() const {
        return feature_;
    }

    void set_feature(FaceFeature feature) {
        feature_ = feature;
    }

 private:
    FaceFeature feature_;
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
