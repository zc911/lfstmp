//
// Created by jiajaichen on 16-6-15.
//

#ifndef PROJECT_STORAGE_REQUEST_H
#define PROJECT_STORAGE_REQUEST_H
#include "witness_bucket.h"
#include "clients/spring_client.h"
#include "clients/data_client.h"
#include "simple_thread_pool.h"

namespace dg {
class StorageRequest {
public:
    StorageRequest(const Config *config) {
        string address = (string) config->Value(STORAGE_ADDRESS);
        spring_client_.CreateConnect(address);
        data_client_.CreateConnect(address);
        pool_ = new ThreadPool(4);
    }

    MatrixError storage() {
        VLOG(VLOG_SERVICE) << "========START REQUEST===========" << endl;

        MatrixError err;
        shared_ptr<WitnessVehicleObj> wv = WitnessBucket::Instance().Pop();
        for (int i = 0; i < wv->storages_size(); i++) {

            string address = wv->storages(i).address();
            if (wv->storages(i).type() == model::POSTGRES) {
                pool_->enqueue([address, this, wv, &err]() {
                    MatrixError errTmp = this->data_client_.SendBatchData(address, wv->mutable_vehicleresult());
                    if (errTmp.code() != 0) {
                        err.set_code(errTmp.code());
                        err.set_message("send to postgres error");
                    }
                });
            } else if (wv->storages(i).type() == model::KAFKA) {
                pool_->enqueue([address, this, wv, &err]() {
                    const VehicleObj &v = wv->vehicleresult();
                    MatrixError errTmp = spring_client_.IndexVehicle(address, v);
                    if (errTmp.code() != 0) {
                        err.set_code(errTmp.code());
                        err.set_message("send to kafka error");
                    }
                });

            }
        }
        return err;
    }
    ~StorageRequest() {
        if (pool_) {
            delete pool_;
        }
    }
private:
    ThreadPool *pool_;
    SpringClient spring_client_;
    DataClient data_client_;
};
}
#endif //PROJECT_STORAGE_REQUEST_H
