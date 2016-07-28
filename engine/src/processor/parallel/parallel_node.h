//
// Created by chenzhen on 7/7/16.
//

#ifndef PROJECT_PARALLEL_PROCESSOR_H_H
#define PROJECT_PARALLEL_PROCESSOR_H_H

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "model/frame.h"
using namespace std;

namespace dg {

class ParallelProcessorNode {
public:
    ParallelProcessorNode() : stop_(true) {
        if (stop_) {
            stop_ = false;
            std::thread t(&ParallelProcessorNode::process, this);
            if (t.joinable())
                t.detach();
        }
    }

    virtual Frame *operator()(Frame *f) = 0;

    void Put(Frame *frame) {
        {
            std::lock_guard<std::mutex> lk(queue_lock_);
            queue_.push(frame);
            if (queue_.size() == 1) {
                queue_empty_cv_.notify_one();
            }
        }
    }

    void process() {
        Frame *f = NULL;

        while (!stop_) {
            {
                if (queue_.empty()) {
                    std::unique_lock<std::mutex> cvlk(queue_lock_);
                    queue_empty_cv_.wait(cvlk, [this] {
                      return (!this->queue_.empty());
                    });
                }

                {
                    std::lock_guard<std::mutex> lk(queue_lock_);
                    f = queue_.front();
                    queue_.pop();
                }

                if (f != NULL) {
                    this->operator()(f);
                    for (auto node : successors_) {
                        node->Put(f);
                    }
                }

            }

        }
    }

    int QueueLoad(){
        return queue_.size();
    }

private:
    volatile bool stop_;
    std::mutex queue_lock_;
    std::mutex cv_lock_;
    std::condition_variable queue_empty_cv_;
    queue<Frame *> queue_;
    vector<ParallelProcessorNode *> successors_;

};

}


#endif //PROJECT_PARALLEL_PROCESSOR_H_H
