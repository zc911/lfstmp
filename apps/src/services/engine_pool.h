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
#include "log/log_val.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <engine/witness_engine.h>
namespace dg {

using namespace std;


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
    VLOG(VLOG_SERVICE) << msg;
    Finish();
  }

  void Run() {

    func();

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

  static MatrixEnginesPool<EngineType> *GetInstance() {
    static MatrixEnginesPool<EngineType> instance;
    return &instance;
  }

  void Run(Config *config) {

    if (!stop_) {
      LOG(ERROR) << "The engine pool already runing" << endl;
      return;
    }
    config_ = config;
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
        cout << "Start thread: " << name<<" "<<engine << endl;

        workers_.emplace_back([this, engine, name] {
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
            LOG(INFO) << name << " " << task;

            // assign the current engine instance to task
            task->apps = (void *) engine;

            // task first binds the engine instance to the specific member methods
            // and then invoke the binded function

            task->Run();


          }
          cout << "end thread: " << name << endl;

          LOG(ERROR) << "Engine thread " << name << " crashed!!!" << endl;
        });
      }

    }
    ModelsMap *modelsMap = ModelsMap::GetInstance();
    modelsMap->clearModels();
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

    }
    condition_.notify_one();
    return true;
  }

private:

  MatrixEnginesPool() : stop_(true) {
  }

  void PrintStastics() {
    VLOG_EVERY_N(VLOG_SERVICE, 100) << endl;
    VLOG_EVERY_N(VLOG_SERVICE, 100) << "========Engine Pool Stastics========" << endl;
    VLOG_EVERY_N(VLOG_SERVICE, 100) << "== Worker number in total: " << workers_.size() << endl;
    VLOG_EVERY_N(VLOG_SERVICE, 100) << "== Task in queue: " << tasks_.size() << endl;
    VLOG_EVERY_N(VLOG_SERVICE, 100) << "========Engine Pool Stastics========" << endl;
    VLOG_EVERY_N(VLOG_SERVICE, 100) << endl;
  }


private:
  Config *config_;
  queue<EngineData *> tasks_;
  vector<std::thread> workers_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;

};

}


#endif //PROJECT_ENGINE_POOL_H
