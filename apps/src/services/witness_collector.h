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
    std::mutex mtx;
    std::condition_variable cv;

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

    void Push(RequestItem * item) {
        unique_lock <mutex> lock(mtx);
        while (max_size_ == tasks_.size())
            not_full.wait(lock);

        tasks_.push(item);
        not_empty.notify_all();
        lock.unlock();
        if(tasks_.size()>=batch_size_&&batch_size_!=1)
          cv_work.notify_all();

    }

    vector<RequestItem* > *Pop() {
        unique_lock <mutex> lock(mtx);

        while (tasks_.size() == 0) {
            not_empty.wait(lock);
        }
        lock.unlock();
        unique_lock <mutex> waitlc(mt_pop);
        vector<RequestItem* > *results = new vector<RequestItem*>();
        if(cv_work.wait_for(waitlc,
                    std::chrono::milliseconds(timeout_),
                    [this]() {return this->batch_size_ <= this->tasks_.size(); })){
        }
        while(tasks_.size()) {
            RequestItem *task = tasks_.front();
            results->push_back(task);
            tasks_.pop();
        }
        not_full.notify_all();
        return results;
    }

    int Size() {

        return tasks_.size();
    }
    void SetBatchsize(int batchsize){
      batch_size_=batchsize;
    }
    void SetTimeout(int timeout){
      timeout_=timeout;
    }

    std::mutex mt_pop;
    std::mutex mtx;
    condition_variable not_full;
    condition_variable not_empty;
    std::condition_variable cv_work;


private:


    WitnessCollector() { };
    WitnessCollector(const WitnessCollector &) { };
    //WitnessBucket &operator=(const WitnessBucket &) { };
    queue<RequestItem* > tasks_;
    int max_size_ = 10;
    int batch_size_ = 8;
    int timeout_ = 100;

};
class WitnessAssembler {
public:
    WitnessAssembler(int worker_size) {

        pool_ = new ThreadPool(worker_size);
    }

    void Run() {
      while(1){
    //    VLOG(VLOG_SERVICE) << "========START REQUEST " << WitnessCollector::Instance().Size() << "===========" << endl;

        vector<RequestItem * > *wv = WitnessCollector::Instance().Pop();
        pool_->enqueue([wv](){
          if(wv->size()>0) {
              FrameBatch framebatch(wv->at(0)->frame->id() * 10);
              for (int i = 0; i < wv->size(); i++) {
                  Frame *frame = wv->at(i)->frame;
                  framebatch.AddFrame(frame,false);
              }

              MatrixEnginesPool <WitnessEngine> *engine_pool = MatrixEnginesPool<WitnessEngine>::GetInstance();

              EngineData data;
              data.func = [&framebatch, &data]() -> void {
                return (bind(&WitnessEngine::Process, (WitnessEngine *) data.apps,
                             placeholders::_1))(&framebatch);
              };

              if (engine_pool == NULL) {
                  LOG(ERROR) << "Engine pool not initailized. " << endl;
                  return ;
              }

              engine_pool->enqueue(&data);
              data.Wait();
              for(int i=0;i<wv->size();i++){
                  wv->at(i)->isFinish=true;
                  wv->at(i)->cv.notify_all();
              }
          }
          delete wv;
        });
      }
    }
    ~WitnessAssembler() {
        if (pool_) {
            delete pool_;
        }
    }
private:
    ThreadPool *pool_;

};
}
#endif //PROJECT_WITNESS_COLLECTOR_H
