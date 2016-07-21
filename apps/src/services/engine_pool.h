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
#include "log/log_val.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <engine/witness_engine.h>
namespace dg {

using namespace std;
using namespace ::dg::model;

class CallData {
public:
  CallData() : finished_(false) {
  }

  MatrixError Wait() {
    std::unique_lock<std::mutex> lock(m_);
    cond_.wait(lock, [this]() { return this->finished_; });
    return result_;
  }
  void Finish() {
    finished_ = true;
    cond_.notify_all();
  }


  void Error(string msg) {
    result_.set_code(-1);
    result_.set_message(msg);
    Finish();
  }

  void Run() {
    result_ = func();
    Finish();
  }


  std::function<MatrixError()> func;
  void *apps;

private:
  volatile bool finished_;
  MatrixError result_;
  std::mutex m_;
  std::condition_variable cond_;

};

class EngineData {
public:
  EngineData() : finished_(false) {
  }

  void Wait() {
    std::unique_lock<std::mutex> lock(m_);
    cond_.wait(lock, [this]() { return this->finished_; });
  }
  void Finish() {
    finished_ = true;
    cond_.notify_all();
  }


  void Error(string msg) {
    VLOG(VLOG_SERVICE)<<msg;
    Finish();
  }

  void Run() { func();
    Finish();
  }


  std::function<void()> func;
  void *apps;

private:
  volatile bool finished_;
  std::mutex m_;
  std::condition_variable cond_;

};
template<typename EngineType>
class MatrixEnginesPool {
public:

    typedef struct {
        int status = 0;
    } WorkerStatus;

    MatrixEnginesPool(Config *config) : config_(config) ,stop_(true){
    }

    void PrintStastics() {
        VLOG_EVERY_N(VLOG_SERVICE, 100) << endl;
        VLOG_EVERY_N(VLOG_SERVICE, 100) << "========Engine Pool Stastics========" << endl;
        VLOG_EVERY_N(VLOG_SERVICE, 100) << "== Worker number in total: " << workers_.size() << endl;
        VLOG_EVERY_N(VLOG_SERVICE, 100) << "== Task in queue: " << tasks_.size() << endl;
        VLOG_EVERY_N(VLOG_SERVICE, 100) << "========Engine Pool Stastics========" << endl;
        VLOG_EVERY_N(VLOG_SERVICE, 100) << endl;
    }

    void Run() {
        if (!stop_) {
            LOG(ERROR) << "The engine pool already runing" << endl;
            return;
        }

        vector<int> threadsOnGpu;
        int gpuNum = config_->Value(SYSTEM_THREADS + "/Size");
        cout << "Gpu num defined in config file: " << gpuNum << endl;

        for (int gpuId = 0; gpuId < gpuNum; ++gpuId) {

            config_->AddEntry("System/GpuId", AnyConversion(gpuId));
            int threadNum = (int) config_->Value(SYSTEM_THREADS + to_string(gpuId));
            cout << "Threads num: " << threadNum << " on GPU: " << gpuId << endl;

            for (int i = 0; i < threadNum; ++i) {
                string name = "engines_" + to_string(gpuId) + "_" + to_string(i);
                EngineType *engine = new EngineType(*config_);
                cout << "Start thread: " << name << endl;

                workers_.emplace_back([this, engine, &name] {
                  for (; ;) {
                      EngineData *task;
                      {

                          std::unique_lock<std::mutex> lock(queue_mutex_);
                          condition_.wait(lock, [this] {
                            return (this->stop_ || !this->tasks_.empty());
                          });

                          if (this->stop_ || this->tasks_.empty()) {
                              continue;
                          }

                          task = this->tasks_.front();
                          this->tasks_.pop();
                          lock.unlock();
                      }
                      // assign the current engine instance to task
                      task->apps = (void *) engine;
                      // task first binds the engine instance to the specific member methods
                      // and then invoke the binded function
                      task->Run();
                  }
                  LOG(ERROR) << "Engine thread " << name << " crashed!!!" << endl;
                });
            }

        }

        cout << "Engine pool worker number: " << workers_.size() << endl;
        stop_ = false;
    }

    bool enqueue(EngineData *data) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                data->Error("Engine pool not running");
                return false;
            }
            tasks_.push(data);
            sleep(100);

        }
        condition_.notify_one();
        return true;
    }

private:
    Config *config_;
    queue<EngineData *> tasks_;
    //  vector<WorkerStatus> worker_status_;
    vector<std::thread> workers_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;

};
    template<class ServiceType,class EngineType >

class ServicePool {
public:

  ServicePool(Config *config,MatrixEnginesPool<EngineType> *engine_pool,int threadNum) : config_(config),engine_pool_(engine_pool) , thread_num_(threadNum) ,stop_(true){

  }

  void Run() {
    if (!stop_) {
      LOG(ERROR) << "The engine pool already runing" << endl;
      return;
    }
    for (int i = 0; i < thread_num_; ++i) {
      string name = "apps_" + to_string(i);
      ServiceType *service = new ServiceType(config_,engine_pool_,name,i);
    VLOG(VLOG_SERVICE)<<"start service "<<name;
      workers_.emplace_back([this, service] {
        for (; ;) {
          CallData *task;
          {

            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] {
              return (this->stop_ || !this->tasks_.empty());
            });

            if (this->stop_ || this->tasks_.empty()) {
              continue;
            }

            task = this->tasks_.front();
            this->tasks_.pop();
            lock.unlock();
          }
          // assign the current engine instance to task
          task->apps = (void *) service;
          // task first binds the engine instance to the specific member methods
          // and then invoke the binded function
          task->Run();
        }
      });

    }

    cout << "Engine pool worker number: " << workers_.size() << endl;
    stop_ = false;
  }

  bool enqueue(CallData *data) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (stop_) {
        data->Error("Engine pool not running");
        return false;
      }
      tasks_.push(data);
    }
    condition_.notify_one();
    return true;
  }


private:
  Config *config_;
  queue<CallData *> tasks_;
  //  vector<WorkerStatus> worker_status_;
  vector<std::thread> workers_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;
  int thread_num_;
    MatrixEnginesPool<EngineType> *engine_pool_;
};

}


#endif //PROJECT_ENGINE_POOL_H
