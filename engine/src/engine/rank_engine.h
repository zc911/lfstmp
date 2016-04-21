/*============================================================================
 * File Name   : rank_engine.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_ENGINE_RANK_ENGINE_H_
#define MATRIX_ENGINE_RANK_ENGINE_H_

#include <glog/logging.h>

#include "engine.h"
#include "timing_profiler.h"
#include "model/frame.h"
#include "model/model.h"
#include "model/ringbuffer.h"
#include "model/rank_feature.h"
#include "processor/car_rank_processor.h"
#include "processor/face_rank_processor.h"


namespace dg {

class RankEngine : public Engine {
public:
    RankEngine(RingBuffer *buffer, Processor *processor)
            : Engine(),
              buffer_(buffer),
              processor_(processor),
              cur_frame_(0)
    {
        assert(buffer_ != NULL);
        assert(processor_ != NULL);
    }
    virtual ~RankEngine() {}

    virtual void Process() override {
        if (buffer_->IsEmpty()) {
            return;
        }

        if (cur_frame_ >= buffer_->Size()) {
            return;
        }

        Frame *f = buffer_->Back();
        if (f == NULL) {
            return;
        }

        cur_frame_ = (cur_frame_ + 1) % buffer_->Size();

        processor_->Update(f);
        return;
    }

    virtual int Stop() {
        return 0;
    }

    virtual int Release() {
        return 0;
    }

 private:
    RingBuffer *buffer_;
    Processor *processor_;
    unsigned int cur_frame_;
};

class CarRankEngine : public RankEngine {
public:
    CarRankEngine() : id_(0), buffer_(5), RankEngine(&buffer_, &processor_) {}
    virtual ~CarRankEngine() {}

    vector<Score> Rank(const Mat& image, const Rect& hotspot, const vector<CarFeature>& candidates)
    {
        vector<Rect> hotspots;
        hotspots.push_back(hotspot);
        CarRankFrame f(id_++, image, hotspots, candidates);
        buffer_.TryPut(&f);
        Process();
        return f.result;
    }

private:
    Identification id_;
    RingBuffer buffer_;
    CarRankProcessor processor_;
};


class FaceRankEngine : public RankEngine {
public:
    FaceRankEngine() : id_(0), buffer_(5), RankEngine(&buffer_, &processor_) {}
    virtual ~FaceRankEngine() {}

    vector<Score> Rank(const Mat& image, const Rect& hotspot, const vector<FaceFeature>& candidates)
    {
        vector<Rect> hotspots;
        hotspots.push_back(hotspot);
        FaceRankFrame f(id_++, image, hotspots, candidates);
        buffer_.TryPut(&f);
        Process();
        return f.result;
    }

private:
    Identification id_;
    RingBuffer buffer_;
    FaceRankProcessor processor_;
};

}

#endif //MATRIX_ENGINE_RANK_ENGINE_H_