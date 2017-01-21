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

    void Run(void *engine) {
        this->apps = engine;
        func();
        Finish();
    }

    void *get_apps() {
        return apps;
    }


    std::function<void()> func;

 private:
    volatile bool finished_;
    void *apps;
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
        unsigned int totalThreadNum = 0;
        std::atomic_int initEngineCount(0);
        std::condition_variable initCv;
        std::mutex initMutex;

        for (int gpuId = 0; gpuId < gpuNum; ++gpuId) {
            int threadNum = (int) config_->Value(SYSTEM_THREADS + to_string(gpuId));
            totalThreadNum += threadNum;
        }

        for (int gpuId = 0; gpuId < gpuNum; ++gpuId) {

            config->AddEntry("System/GpuId", AnyConversion(gpuId));
            int threadNum = (int) config_->Value(SYSTEM_THREADS + to_string(gpuId));
            cout << "Threads num: " << threadNum << " on GPU: " << gpuId << endl;

            // copy the config instance because it will be changed by another thread
            Config configCopy = *config;
            std::mutex initMutex;
            for (int i = 0; i < threadNum; ++i) {
                string name = "engines_" + to_string(gpuId) + "_" + to_string(i);

                workers_.emplace_back([this, name, configCopy, &initEngineCount, totalThreadNum, &initCv, &initMutex] {
                    cout << "Start: " << name  << " at gpu " << (int)configCopy.Value("System/GpuId") << " and thread no.: "
                        << std::this_thread::get_id() << endl;

                    EngineType *engine = nullptr;
                    // since dgface fcn detection must be init in seperated threads, so we init the engine
                    // in threads. But the engine init is not thread safe, so we init the engines one by one.
                    {
                        std::lock_guard<std::mutex> initLock(initMutex);
                        engine = new EngineType(configCopy);
                    }

                    if (engine == nullptr){
                        LOG(FATAL) << "Init engine error" << endl;
                    }

                    initEngineCount++;
                    if (initEngineCount == totalThreadNum) {
                        initCv.notify_all();
                    }


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
                        VLOG(VLOG_RUNTIME_DEBUG) << name << " " << task;

                        // assign the current engine instance to task
                        // task first binds the engine instance to the specific member methods
                        // and then invoke the binded function
                        VLOG(VLOG_RUNTIME_DEBUG)
                        << "Process at engine: " << name << " at thread: " << std::this_thread::get_id() << endl;
                        task->Run((void *) engine);

                    }
                    cout << "end thread: " << name << endl;
                    LOG(ERROR) << "Engine thread " << name << " crashed!!!" << endl;
                });
            }

        }

        // make sure all threads are init successfully
        std::unique_lock<std::mutex> lck(initMutex);
        if (initCv.wait_for(lck, std::chrono::seconds(1200), [&]() {
            return totalThreadNum == initEngineCount;
        })) {
            cout << "All engine threads init successful, thread number: " << initEngineCount << endl;
        } else {
            LOG(FATAL) << "Engine threads init timeout or error, init count " << initEngineCount << " but should be "
                << totalThreadNum << endl;
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
