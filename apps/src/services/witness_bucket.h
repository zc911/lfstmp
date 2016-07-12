//
// Created by jiajaichen on 16-6-15.
//

#ifndef PROJECT_WITNESS_BUCKET_H
#define PROJECT_WITNESS_BUCKET_H
#include "model/spring.grpc.pb.h"
#include "model/witness.grpc.pb.h"
#include "glog/logging.h"
#include "log/log_val.h"
using namespace std;
using namespace ::dg::model;
using ::dg::model::SpringService;
using ::dg::model::WitnessService;
namespace dg {
class WitnessBucket {
public:
    ~WitnessBucket() { };
    static WitnessBucket &Instance() {
        return instance_;
    }
    void SetMaxSize(int num) {
        max_size_ = num;
    }
    void Push(shared_ptr<WitnessVehicleObj> item) {
        unique_lock<mutex> lock(mtx);
        while (max_size_ == tasks.size())
            not_full.wait(lock);
        VLOG(VLOG_SERVICE) << "tasks size " << tasks.size() << endl;
        tasks.push(item);
        not_empty.notify_all();
        lock.unlock();
    }
    shared_ptr<WitnessVehicleObj> Pop() {
        unique_lock<mutex> lock(mtx);
        while (tasks.size() == 0) {
            not_empty.wait(lock);
        }
        shared_ptr<WitnessVehicleObj> task = tasks.front();
        tasks.pop();
        not_full.notify_all();
        lock.unlock();
        return task;
    }
    int Size() {
        return tasks.size();
    }
    std::mutex mt_pop;
    std::mutex mt_push;
    std::mutex mtx;
    condition_variable not_full;
    condition_variable not_empty;
private:
    WitnessBucket() { };
    WitnessBucket(const WitnessBucket &) { };
    WitnessBucket &operator=(const WitnessBucket &) { };
    queue<shared_ptr<WitnessVehicleObj> > tasks;
    int max_size_ = 10;
    int current_ = 0;
    static WitnessBucket instance_;

};
}
#endif //PROJECT_WITNESS_BUCKET_H
