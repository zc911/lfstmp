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
#include "dgface/alignment.h"

using namespace std;
using namespace cv;
namespace dg {

typedef enum {
    OBJECT_UNKNOWN = 0,
    OBJECT_CAR = 1,
    OBJECT_PEDESTRIAN = 2,
    OBJECT_BICYCLE = 4,
    OBJECT_TRICYCLE = 8,
    OBJECT_WINDOW = 16,
    OBJECT_MARKER = 32,
    OBJECT_DRIVER = 64,
    OBJECT_CODRIVER = 128,
    OBJECT_FACE = 256,
} ObjectType;

enum DetectionTypeId {
    DETECTION_UNKNOWN = 0,
    DETECTION_CAR = 1,
    DETECTION_PEDESTRIAN = 2,
    DETECTION_BICYCLE = 3,
    DETECTION_TRICYCLE = 4,
    DETECTION_FACE = 5
};


typedef struct Detection {
 public:
    int id = -1;
    bool deleted;
    Confidence confidence = 0;
    float col_ratio = 1.0;
    float row_ratio = 1.0;

    void Rescale(float scale) {
        box_.x = box_.x / scale;
        box_.y = box_.y / scale;
        box_.width = box_.width / scale;
        box_.height = box_.height / scale;
    }
    friend ostream &operator<<(std::ostream &os, const Detection &det) {
//        return os << "DETECTION_ID: " << det.id << " BOX: [" << det.box().x << ","
//            << det.box().y << "," << det.box().width << "," << det.box().height
//            << "] Conf: " << det.confidence;
        return os;
    }
    void set_rotated_box(const RotatedBox &rbox) {
        rotated_box_ = rbox;
        box_ = rbox.boundingRect();
    }
    void set_box(const Box &bbox) {
        Point2f center = Point2f(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
        Size2f size = Size2f(bbox.width, bbox.height);
        rotated_box_ = cv::RotatedRect(center, size, 0);
        box_ = bbox;
    }
    Box box() const {
        return box_;
    }
    RotatedBox rotated_box() const {
        return rotated_box_;
    }

 private:
    Box box_;
    RotatedBox rotated_box_;

} Detection;

typedef struct FacePose {
    int type;
    vector<float> angles;
} FacePose;

class Object {
 public:

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

    virtual void set_image(const cv::Mat &image) {
        // empty implements
    };

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
    vector<Object *> children(ObjectType type) const {
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

    enum { NoBelt = 48, Phone = 47 };
    enum { Yes = 1, No = 0, NotSure = 2, NoPerson = 3 };
    Vehicler(ObjectType type) : Object(type) {

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
    Window(Mat &img, vector<Rect> &fobbiden, vector<float> &params)
        : Object(OBJECT_WINDOW), image_(img), fobbiden_(fobbiden), params_(params), class_id_(-1) {
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
            Marker *m = new Marker(OBJECT_MARKER);
            m->set_detection(detection);
            m->set_class_id(detection.id);
            m->set_confidence(detection.confidence);
            this->AddChild(m);
        }
    }
    vector<Rect> &fobbiden() {
        return fobbiden_;
    }
    vector<float> &params() {
        return params_;
    }
    Mat &resized_image() {
        return resized_image_;
    }
    Mat &phone_image() {
        return phone_image_;
    }
    Mat &image() {
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

    typedef enum {
        VEHICLE_POSE_HEAD = 1,
        VEHICLE_POSE_TAIL = 2
    } VehiclePose;

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
    void set_window(Window *window) {
        this->AddChild(window);
    }
    void set_vehicler(Vehicler *vehicler) {
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

    const VehiclePose pose() const {
        return pose_;
    }

    void set_pose(VehiclePose pose) {
        pose_ = pose;
    }

 private:

    cv::Mat image_;
    cv::Mat resized_image_;
    Identification class_id_;
    vector<Plate> plates_;
    Color color_;
    CarRankFeature feature_;
    VehiclePose pose_;

};

class NonMotorVehicle: public Object {
 public:
    typedef struct {
        int index = 0;
        string tagname = "";
        Confidence confidence = 0;
        float threshold_lower = 0;
        float threshold_upper = 0;
        int mappingId = 0;
        int categoryId = 0;
    } Attr;

    NonMotorVehicle(ObjectType type) : Object(type) {
    }

    ~NonMotorVehicle() {

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

    map<string, float> &threshold() {
        return threshold_;
    };

    vector<Attr> &attrs() {
        return attrs_;
    }

 private:

    cv::Mat image_;
    cv::Mat resized_image_;
    std::vector<Attr> attrs_;
    std::map<string, float> threshold_;

};

class Face: public Object {

 public:
 public:
    enum { BlurM = 0, Frontal = 1 };
    enum { NotFrontalType = 1, FrontalType = 0 };
    const float Pitch = 30;
    const float Yaw = 30;
    Face()
        : Object(OBJECT_FACE), is_valid_(true) {

    }

    Face(Identification id, Detection detection, Confidence confidence)
        : Object(OBJECT_FACE), is_valid_(true) {
        id_ = id;
        confidence_ = confidence;
        detection_ = detection;
    }

    Face(Identification id, int x, int y, int width, int height,
         Confidence confidence)
        : Object(OBJECT_FACE), is_valid_(true) {
        id_ = id;
        confidence_ = confidence;
        detection_.set_box(Box(x, y, width, height));
    }

    FaceRankFeature feature() const {
        return feature_;
    }

    void set_feature(FaceRankFeature feature) {
        feature_ = feature;
    }

    const cv::Mat &full_image() const {
        return full_image_;
    }

    void set_full_image(const cv::Mat &image) {
        full_image_ = image;
    }

    const cv::Mat &image() const {
        return image_;
    }

    void set_image(const cv::Mat &image) {
        image_ = image;
    }

    void set_image(const Detection detection) {
        Box newBox = detection.box();
        if(detection.box().x + detection.box().width >= full_image_.cols){
            newBox.width = full_image_.cols - detection.box().x;
        }
        if(detection.box().y + detection.box().height >= full_image_.rows){
            newBox.height = full_image_.rows - detection.box().y;
        }
        image_ = full_image_(newBox);
    }


    bool IsValid() {
        return is_valid_;
    }
    void set_valid(bool flag) {
        is_valid_ = flag;
    }

    void set_qualities(int type, float score) {
        qualities_[type] = score;
    }
    const map<int, float> &get_qualities() const {
        return qualities_;
    }
    void set_pose(vector<float> angles) {
        face_pose_.angles = angles;
        if ((abs(angles[0]) <= Pitch) || (abs(angles[1]) <= Yaw)) {
            face_pose_.type = FrontalType;
        } else {
            face_pose_.type = NotFrontalType;
        }
    }
    FacePose get_pose() const {
        return face_pose_;
    }
    void set_align_result(DGFace::AlignResult align_result) {
        align_result_ = align_result;
    }
    const DGFace::AlignResult &get_align_result() const {
        return align_result_;
    }
 private:
    cv::Mat full_image_;
    cv::Mat image_;
    FaceRankFeature feature_;
    bool is_valid_;
    map<int, float> qualities_;
    FacePose face_pose_;
    DGFace::AlignResult align_result_;
};

}
#endif /* MODEL_H_ */
