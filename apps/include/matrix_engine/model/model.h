/*
 * model.h
 *
 *  Created on: 01/04/2016
 *      Author: chenzhen
 */

#ifndef MODEL_H_
#define MODEL_H_

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "basic.h"
#include "rank_feature.h"

using namespace std;
using namespace cv;
namespace dg {

typedef enum {
    OBJECT_UNKNOWN = 0,
    OBJECT_CAR = 1,
    OBJECT_PEDESTRIAN = 2,
    OBJECT_BICYCLE = 3,
    OBJECT_TRICYCLE = 4,
    OBJECT_WINDOW = 8,
    OBJECT_MARKER = 16,
    OBJECT_MARKER_0 = 16,
    OBJECT_MARKER_1 = 16,
    OBJECT_MARKER_2 = 16,
    OBJECT_MARKER_3 = 16,
    OBJECT_MARKER_4 = 16,
    OBJECT_PEOPLE = 32,
    OBJECT_DRIVER = 32,
    OBJECT_CODRIVER = 64,
    OBJECT_FACE = 128,
} ObjectType;

enum DetectionTypeId {
    DETECTION_UNKNOWN = 0,
    DETECTION_CAR = 1,
    DETECTION_PEDESTRIAN = 2,
    DETECTION_BICYCLE = 3,
    DETECTION_TRICYCLE = 4
};



typedef struct Detection {
    int id = -1;
    bool deleted;
    Box box;
    Confidence confidence = 0;
    float col_ratio = 1.0;
    float row_ratio = 1.0;
    void Rescale(float scale) {
        box.x = box.x / scale;
        box.y = box.y / scale;
        box.width = box.width / scale;
        box.height = box.height / scale;
    }
    friend ostream &operator<<(std::ostream &os, const Detection &det) {
        return os << "DETECTION_ID: " << det.id << " BOX: [" << det.box.x << ","
               << det.box.y << "," << det.box.width << "," << det.box.height
               << "] Conf: " << det.confidence;
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
            Object *obj = children_[i];
            if (obj) {
                delete obj;
                obj = NULL;
            }
        }
        children_.clear();
    }

    const vector<Object *> &children() const {
        return children_;
    }

    void AddChild(Object *child) {
        children_.push_back(child);
    }

    const Detection &detection() const {
        return detection_;
    }

    void set_detection(const Detection &detection) {
        detection_ = detection;
    }
    Object *child(ObjectType type) const {
        for (int i = 0; i < children_.size(); i++) {

            if (children_[i]->type() == type) {
                return children_[i];
            }
        }
        return NULL;

    }
    vector<Object *>children(ObjectType type)const {
        vector<Object *> result;
        for (auto *child : children_) {
            if (child->type() == type)
                result.push_back(child);
        }
        return result;
    }

    Identification id() const {
        return id_;
    }

    void set_id(Identification id) {
        id_ = id;
    }

    Confidence confidence() const {
        return confidence_;
    }

    void set_confidence(Confidence confidence) {
        confidence_ = confidence;
    }

    ObjectType type() const {
        return type_;
    }

    void set_type(ObjectType type) {
        type_ = type;
    }

protected:
    Identification id_;
    Confidence confidence_;
    ObjectType type_;
    Detection detection_;
    vector<Object *> children_;

};
class Vehicler: public Object {
public:

    enum {NoBelt = 48, Phone = 47};
    enum {Yes = 1, No = 0, NotSure = 2, NoPerson = 3};
    Vehicler(ObjectType type): Object(type) {

    }
    ~Vehicler() {

    }
    void set_vehicler_attr(int key, float value) {
        vehicler_attr_.insert(pair<int, float>(key, value));
    }
    float vehicler_attr_value(int key) {
        map<int, float>::iterator it = vehicler_attr_.find(key);
        if (it != vehicler_attr_.end()) {
            return it->second;
        }
        return 0;
    }
    map<int, float> vehicler_attr_;
};

class Marker: public Object {
public:
    Marker(ObjectType type)
        : Object(type),
          class_id_(-1) {
    }
    ~Marker() {

    }
    Identification class_id() const {
        return class_id_;
    }

    void set_class_id(Identification id) {
        class_id_ = id;
    }

private:
    Identification class_id_;

};
typedef struct {
    Identification id;
    Timestamp timestamp;
    MessageStatus status;
    MetaData *video_meta_data;
    Object *object;
} Message;
class Window: public Object {
public:
    Window(Mat &img, vector<Rect> &fobbiden, vector<float> &params): Object(OBJECT_WINDOW), image_(img), fobbiden_(fobbiden), params_(params), class_id_(-1) {
        params_.resize(6);
    }
    ~Window() {

    }
    void set_resized_img(Mat &img) {
        resized_image_ = img;
    }
    void set_phone_img(Mat &img) {
        phone_image_ = img;
    }
    Identification class_id() const {
        return class_id_;
    }

    void set_class_id(Identification id) {
        class_id_ = id;
    }
    void set_markers(const vector<Detection> &markers) {

        for (auto detection : markers) {
            Marker *m = new Marker(OBJECT_MARKER_0);
            m->set_detection(detection);
            m->set_class_id(detection.id);
            m->set_confidence(detection.confidence);
            this->AddChild(m);
        }
    }
    vector<Rect> & fobbiden() {
        return fobbiden_;
    }
    vector<float> & params() {
        return params_;
    }
    Mat & resized_image() {
        return resized_image_;
    }
    Mat & phone_image() {
        return phone_image_;
    }
    Mat & image() {
        return image_;
    }
private:
    Identification class_id_;
    Mat resized_image_;
    Mat image_;
    Mat phone_image_;
    vector<cv::Rect> fobbiden_;
    vector<float> params_;
};
class Pedestrian: public Object {
public:
    typedef struct {
        int index = 0;
        string tagname = "";
        Confidence confidence = 0;
        float threshold_lower = 0;
        float threshold_upper = 0;
        int categoryId = 0;
        int mappingId = 0;
    } Attr;

    Pedestrian() : Object(OBJECT_PEDESTRIAN) {
    }
    cv::Mat &image() {
        return image_;
    }
    void set_image(const cv::Mat &image) {
        image_ = image;
    }

    const std::vector<Attr> &attrs() const {
        return attrs_;
    }

    void set_attrs(const std::vector<Attr> &attrs) {
        attrs_ = attrs;
    }

    const std::map<string, float> &threshold() const {
        return threshold_;
    }

    void set_threshold(const std::map<string, float> &threshold) {
        threshold_ = threshold;
    }

private:
    cv::Mat image_;
    std::vector<Attr> attrs_;
    std::map<string, float> threshold_;
};

class Vehicle: public Object {
public:

    typedef struct {
        Identification class_id = -1;
        Confidence confidence = 0;
    } Color;

    typedef struct {
        Box box;
        string plate_num = "";
        int color_id = -1;
        int plate_type = -1;
        Confidence confidence = 0;
        Confidence local_province_confidence = 0;
    } Plate;

    Vehicle(ObjectType type)
        : Object(type),
          class_id_(-1) {
        Plate plate;
        plates_.push_back(plate);
    }

    const Color &color() const {
        return color_;
    }

    void set_color(const Color &color) {
        color_ = color;
    }
    void set_window( Window *window) {
        this->AddChild(window);
    }
    void set_vehicler( Vehicler *vehicler) {
        this->AddChild(vehicler);

    }

    const cv::Mat &image() const {
        return image_;
    }
    const cv::Mat &resized_image() const {
        return resized_image_;
    }
    void set_image(const cv::Mat &image) {
        image_ = image;
        cv::resize(image_, resized_image_, cv::Size(256, 256));
        resized_image_ = resized_image_(cv::Rect(8, 8, 240, 240));
    }

    const vector<Plate> &plates() const {
        return plates_;
    }

    void set_plates(const vector<Plate> &plates) {
        plates_ = plates;
    }

    Identification class_id() const {
        return class_id_;
    }

    void set_class_id(Identification classId) {
        class_id_ = classId;
    }
    const CarRankFeature &feature() const {
        return feature_;
    }

    void set_feature(const CarRankFeature &feature) {
        feature_ = feature;
    }
private:

    cv::Mat image_;
    cv::Mat resized_image_;
    Identification class_id_;
    vector<Plate> plates_;
    Color color_;
    CarRankFeature feature_;

};

class Face: public Object {

public:
    Face()
        : Object(OBJECT_FACE) {

    }

    Face(Identification id, Detection detection, Confidence confidence)
        : Object(OBJECT_FACE) {
        id_ = id;
        confidence_ = confidence;
        detection_ = detection;
    }

    Face(Identification id, int x, int y, int width, int height,
         Confidence confidence)
        : Object(OBJECT_FACE) {
        id_ = id;
        confidence_ = confidence;
        detection_.box = Box(x, y, width, height);
    }

    FaceRankFeature feature() const {
        return feature_;
    }

    void set_feature(FaceRankFeature feature) {
        feature_ = feature;
    }

    const cv::Mat &image() const {
        return image_;
    }

    void set_image(const cv::Mat &image) {
        image_ = image;
    }

private:
    cv::Mat image_;
    FaceRankFeature feature_;
};


}
#endif /* MODEL_H_ */
