/*
 * Frame.h
 *
 *  Created on: 01/04/2016
 *      Author: chenzhen
 */

#ifndef FRAME_H_
#define FRAME_H_

#include <vector>
#include <pthread.h>

#include "payload.h"
#include "rank_feature.h"

using namespace std;
using namespace cv;

namespace dg {

typedef enum {

} FrameType;

enum FrameStatus {
    FRAME_STATUS_INIT = 0,
    FRAME_STATUS_DETECTED = 1,
    FRAME_STATUS_FINISHED = 128
};

class Frame {
 public:
    Frame(const Identification id)
            : id_(id),
              timestamp_(0),
              status_(FRAME_STATUS_INIT),
              operation_(0),
              payload_(0) {

    }
    Frame(const Identification id, unsigned int width, unsigned int height,
          unsigned char *data)
            : id_(id),
              timestamp_(0),
              status_(FRAME_STATUS_INIT),
              operation_(0) {
        payload_ = new Payload(id_, width, height, data);
    }
    virtual ~Frame() {
        if (payload_)
            delete payload_;
    }

    Identification id() const {
        return id_;
    }

    void set_id(Identification id) {
        id_ = id;
    }

    const vector<Object*>& objects() const {
        return objects_;
    }

    void put_object(Object *obj) {
        for (vector<Object *>::iterator itr = objects_.begin();
                itr != objects_.end(); ++itr) {
            Object *old_obj = *itr;

            if (old_obj->id() == obj->id()) {
                delete old_obj;
                itr = objects_.erase(itr);
                break;
            }
        }
        objects_.push_back(obj);
    }

    Object* get_object(Identification id) {
        for (vector<Object *>::iterator itr = objects_.begin();
                itr != objects_.end(); ++itr) {
            Object *obj = *itr;
            if (obj->id() == id) {
                return *itr;
            }
        }
        return NULL;
    }

    void set_objects(const vector<Object*>& objects) {
        objects_ = objects;
    }

    Operation operation() const {
        return operation_;
    }

    void set_operation(Operation operation) {
        operation_ = operation;
    }

    Payload* payload() const {
        return payload_;
    }

    void set_payload(Payload* payload) {
        payload_ = payload;
    }

    volatile FrameStatus status() const {
        return status_;
    }

    void set_status(volatile FrameStatus status) {
        status_ = status;
    }

    Timestamp timestamp() const {
        return timestamp_;
    }

    void set_timestamp(Timestamp timestamp) {
        timestamp_ = timestamp;
    }

 protected:
    Identification id_;
    Timestamp timestamp_;
    volatile FrameStatus status_;
    Operation operation_;
    Payload *payload_;
    // base pointer
    vector<Object *> objects_;
};

class RenderableFrame : public Frame {
 public:
    RenderableFrame();
    ~RenderableFrame();
 private:
    cv::Mat render_data_;
};

// just derive the base class
class FrameBatch : private Frame {
 public:
    FrameBatch();
    ~FrameBatch();
 private:
    Identification id_;
    unsigned int batch_size_;
    vector<Frame *> frames_;
};

class CarRankFrame : public Frame {
public:
    CarRankFrame(Identification id, const Mat& image, const vector<Rect>& hotspots, const vector<CarFeature>& candidates)
            : Frame(id)
            , image(image)
            , hotspots(hotspots)
            , candidates(candidates)
    {}
    ~CarRankFrame(){}
    CarRankFrame(const CarRankFrame& f) 
            : Frame(f.id_)
            , image(f.image)
            , hotspots(f.hotspots)
            , candidates(f.candidates)
    {
    }

    const Mat& image;
    const vector<Rect>& hotspots;
    const vector<CarFeature>& candidates;

    vector<Score> result;
};


class FaceRankFrame : public Frame {
public:
    FaceRankFrame(Identification id, const Mat& image, const vector<Rect>& hotspots, const vector<FaceFeature>& candidates)
            : Frame(id)
            , image(image)
            , hotspots(hotspots)
            , candidates(candidates)
    {}
    ~FaceRankFrame(){}
    FaceRankFrame(const FaceRankFrame& f) 
            : Frame(f.id_)
            , image(f.image)
            , hotspots(f.hotspots)
            , candidates(f.candidates)
    {
    }

    const Mat& image;
    const vector<Rect>& hotspots;
    const vector<FaceFeature>& candidates;

    vector<Score> result;
};


// TODO
class RankData : public Frame {
    Frame *frame_;
    vector<Box> hotspots_;
    vector<Feature> features_;
};

}

#endif /* FRAME_H_ */
