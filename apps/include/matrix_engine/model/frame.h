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

/// Frame represents a single request. A frame encapsulates a payload
/// which is the data will be computed and processed. The processed data
/// will also be found from this class.
class Frame {
 public:

    Frame(const Identification id)
            : id_(id),
              timestamp_(0),
              status_(FRAME_STATUS_INIT),
              payload_(0) {

    }

    /// @param id The frame id which need to be unique
    /// @param width The width of the data
    /// @param height The height of the data
    /// @param data the data
    Frame(const Identification id, unsigned int width, unsigned int height,
          unsigned char *data)
            : id_(id),
              timestamp_(0),
              status_(FRAME_STATUS_INIT) {
        payload_ = new Payload(id_, width, height, data);
    }

    /// @param id The frame id which need to be unique
    /// @param img The data
    Frame(const Identification id, Mat img)
            : id_(id),
              timestamp_(0),
              status_(FRAME_STATUS_INIT) {
        payload_ = new Payload(id_, img);
    }

    virtual ~Frame() {
        if (payload_)
            delete payload_;
        for (int i = 0; i < objects_.size(); ++i) {
            Object * obj = objects_[i];
            if (obj) {
                delete obj;
                obj = NULL;
            }
        }
        objects_.clear();
    }

    Identification id() const {
        return id_;
    }

    void set_id(Identification id) {
        id_ = id;
    }

    vector<Object*>& objects() {
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

    void set_objects(vector<Object*>& objects) {
        objects_ = objects;
    }

    Operation operation() const {
        return operation_;
    }

    void set_operation(Operation operation) {
        operation_ = operation;
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

    int get_object_size() {
        return objects_.size();
    }

    const string& error_Msg() const {
        return error_msg_;
    }

    void set_error_msg(const string& errorMsg) {
        error_msg_ = errorMsg;
    }

    Payload * payload() {
        return payload_;
    }

 protected:
    Identification id_;
    Timestamp timestamp_;
    volatile FrameStatus status_;
    Operation operation_;
    Payload *payload_;
    vector<Object *> objects_;
    string error_msg_;
}
;

class RenderableFrame : public Frame {
 public:
    RenderableFrame();
    ~RenderableFrame();
 private:
    cv::Mat render_data_;
};

// just derive the base class
class FrameBatch {
 public:
    FrameBatch(const Identification id, int batch_size)
            : id_(id),
              batch_size_(batch_size) {

    }
    ~FrameBatch() {
        for (int i = 0; i < frames_.size(); ++i) {
            Frame * f = frames_[i];
            if (f) {
                delete f;
                f = NULL;
            }
        }
        frames_.clear();
    }

    int add_frame(Frame *frame) {
        if (frames_.size() < batch_size_) {
            frames_.push_back(frame);
            return frames_.size();
        } else {
            return -1;
        }
    }
    int add_frames(vector<Frame *> frames) {
        if ((frames.size() + frames_.size()) > batch_size_) {
            return -1;
        } else {
            frames_.insert(frames_.end(), frames.begin(), frames.end());
            return 1;
        }
    }
    vector<Frame *> frames() {
        return frames_;
    }
    unsigned int batch_size() const {
        return batch_size_;
    }
    vector<Object*> CollectObjects() {

        vector<Object *> objects;
        for (auto * frame : frames_) {

            objects.insert(objects.end(), frame->objects().begin(),
                           frame->objects().end());
        }
        return objects;
    }
    vector<Object*> CollectObjects(uint64_t operation) {

        vector<Object *> objects;
        for (auto * frame : frames_) {
            if (!frame->operation().Check(operation))
                continue;
            objects.insert(objects.end(), frame->objects().begin(),
                           frame->objects().end());
        }
        return objects;
    }

    /// Check the operations of each frame.
    /// Return true if any frame satisfy the input operations
    /// Return false otherwise
    bool CheckFrameBatchOperation(OperationValue operations) const {
        for (auto * frame : frames_) {
            if (frame->operation().Check(operations))
                return true;
        }
        return false;
    }

 private:
    Identification id_;
    unsigned int batch_size_;
    vector<Frame *> frames_;
};

class CarRankFrame : public Frame {
 public:
    CarRankFrame(Identification id, const Mat& image,
                 const vector<Rect>& hotspots,
                 const vector<CarRankFeature>& candidates)
            : Frame(id),
              image_(image),
              hotspots_(hotspots),
              candidates_(candidates) {
    }
    ~CarRankFrame() {
    }
    CarRankFrame(const CarRankFrame& f)
            : Frame(f.id_),
              image_(f.image_),
              hotspots_(f.hotspots_),
              candidates_(f.candidates_) {
    }

    const Mat& image_;
    const vector<Rect>& hotspots_;
    const vector<CarRankFeature>& candidates_;

    vector<Score> result_;
};

class FaceRankFrame : public Frame {
 public:
    FaceRankFrame(Identification id, const FaceRankFeature& datum,
                  const vector<Rect>& hotspots,
                  const vector<FaceRankFeature>& candidates)
            : Frame(id),
              datum_(datum),
              hotspots_(hotspots),
              candidates_(candidates) {
    }

    ~FaceRankFrame() {
    }

    const vector<Rect>& hotspots_;
    const vector<FaceRankFeature>& candidates_;
    vector<Score> result_;
    const FaceRankFeature& datum_;
};

}

#endif /* FRAME_H_ */
