//
// Created by chenzhen on 6/2/16.
//

#ifndef PROJECT_ENGINE_POOL_H
#define PROJECT_ENGINE_POOL_H

#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include "../model/common.pb.h"

namespace dg {

using namespace std;
using namespace ::dg::model;

class CallData {
public:
    CallData() {
        finished = false;
    }
    MatrixError Wait() {
        std::unique_lock<std::mutex> lock(m);
        cond.wait(lock, [this]() { return this->finished; });
        return result;
    }

    void Run() {
        result = func();
        finished = true;
        cond.notify_all();
    }

    void *apps() {
        return this->apps_;
    }

    void registApps(void *a) {
        this->apps_ = a;
    }

    std::function<MatrixError()> func;
    void *apps_;
private:
    bool finished;
    MatrixError result;

    std::mutex m;
    std::condition_variable cond;

};

template<typename E>
class MatrixEnginesPool {
public:

    MatrixEnginesPool(Config *config) : config_(config) {

    }

    void Run() {
        if (!stop_) {
            LOG(ERROR) << "The engine pool already runing" << endl;
            return;
        }

        int threadNum = (int) config_->Value("System/ThreadsPerGpu");
        threadNum = threadNum == 0 ? 3 : threadNum;
        cout << "start thread : " << threadNum << endl;
        for (int i = 0; i < threadNum; ++i) {
            WitnessAppsService *engine = new WitnessAppsService(config_, "apps_" + to_string(i));
            workers_.emplace_back([this, engine] {
              for (; ;) {
                  CallData *task;
                  {

                      std::unique_lock<std::mutex> lock(queue_mutex_);
                      condition_.wait(lock, [this] {
                        return (this->stop_ || !this->tasks_.empty());
                      });

                      if (this->stop_ || this->tasks_.empty())
                          return;

                      task = this->tasks_.front();
                      this->tasks_.pop();
                      lock.unlock();
                  }
                  cout << "Process in thread: " << std::this_thread::get_id() << endl;
                  // assign the current engine instance to task
                  task->apps_ = (void *) engine;
                  // task first binds the engine instance to the specific member methods
                  // and then invoke the binded function
                  task->Run();
                  cout << "finish process: " << endl;
              }
            });
        }


        stop_ = false;
    }
    int enqueue(CallData *data) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                cout << "is stop" << endl;
                return 1;
            }
            tasks_.push(data);
        }
        condition_.notify_one();
        return 1;
    }
private:
    Config *config_;
    queue<CallData *> tasks_;
    vector<std::thread> workers_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;

};

}


#endif //PROJECT_ENGINE_POOL_H
