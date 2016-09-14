//
// Created by jiajaichen on 16-9-14.
//

#ifndef PROJECT_WITNESS_COLLECTOR_H
#define PROJECT_WITNESS_COLLECTOR_H

#include <memory>
#include <vector>
#include <queue>
#include "glog/logging.h"
#include "log/log_val.h"
#include "matrix_engine/engine/witness_engine.h"

using namespace std;
namespace dg {
typedef struct {
    Frame *frame;
    bool isFinish=false;
} RequestItem;
class WitnessCollector {
public:
    ~WitnessCollector() { };
    static WitnessCollector &Instance() {
        static WitnessCollector instance;
        return instance;
    }

    void SetMaxSize(int num) {
        max_size_ = num;
    }

    void Push(shared_ptr<RequestItem> item) {
        unique_lock <mutex> lock(mtx);
        while (max_size_ == tasks_.size())
            not_full.wait(lock);
        queue<shared_ptr<RequestItem> > tasks1;

        tasks_.push(item);
        not_empty.notify_all();
        lock.unlock();
    }

    vector<shared_ptr<RequestItem> > Pop() {
        unique_lock <mutex> lock(mtx);
        while (tasks_.size() == 0) {
            not_empty.wait(lock);
        }
        unique_lock <mutex> waitlc(mtx);
        std::condition_variable cv;
        vector<shared_ptr<RequestItem> > results;
        cv.wait_for(waitlc,
                    std::chrono::microseconds(timeout_),
                    [&tasks_, &batch_size_]() { return batch_size_ <= tasks_.size(); });
        for (int i = 0; i < batch_size_; i++) {
            shared_ptr<RequestItem> task = tasks_.front();
            tasks_.pop();
            results.push_back(task);
        }
        not_full.notify_all();
        return results;
    }

    int Size() {
        return tasks_.size();
    }

    std::mutex mt_pop;
    std::mutex mt_push;
    std::mutex mtx;
    condition_variable not_full;
    condition_variable not_empty;

private:


    WitnessCollector() { };
    WitnessCollector(const WitnessCollector &) { };
    //WitnessBucket &operator=(const WitnessBucket &) { };
    queue<shared_ptr<RequestItem> > tasks_;
    int max_size_ = 10;
    int current_ = 0;
    int batch_size_ = 1;
    int timeout_ = 100;

};
class WitnessAssembler {
public:
    WitnessAssembler(int worker_size) {

        pool_ = new ThreadPool(worker_size);
    }

    void batchRecognize() {

        VLOG(VLOG_SERVICE) << "========START REQUEST " << WitnessCollector::Instance().Size() << "===========" << endl;

        vector<shared_ptr<RequestItem> > wv = WitnessCollector::Instance().Pop();
        pool_->enqueue([&wv](){
          if(wv.size()>0) {
              FrameBatch framebatch(wv[0]->frame->id() * 10);
              for (int i = 0; i < wv.size(); i++) {
                  Frame *frame = wv[i]->frame;
                  framebatch.AddFrame(frame);
              }

              MatrixEnginesPool <WitnessEngine> *engine_pool = MatrixEnginesPool<WitnessEngine>::GetInstance();

              EngineData data;
              data.func = [&framebatch, &data]() -> void {
                return (bind(&WitnessEngine::Process, (WitnessEngine *) data.apps,
                             placeholders::_1))(&framebatch);
              };

              if (engine_pool == NULL) {
                  LOG(ERROR) << "Engine pool not initailized. " << endl;
                  return err;
              }

              engine_pool->enqueue(&data);
              gettimeofday(&start, NULL);

              data.Wait();
              for(int i=0;i<wv.size();i++){
                  wv[i]->isFinish=true;
              }
          }
        });

    }
    ~StorageRequest() {
        if (pool_) {
            delete pool_;
        }
    }
private:
    ThreadPool *pool_;

};
}
#endif //PROJECT_WITNESS_COLLECTOR_H
