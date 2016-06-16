//
// Created by jiajaichen on 16-6-6.
//

#ifndef MATRIX_APPS_SPRING_GRPC_H
#define MATRIX_APPS_SPRING_GRPC_H

#include <grpc++/grpc++.h>
#include "../model/witness.grpc.pb.h"
#include "../model/spring.grpc.pb.h"
#include "../services/engine_pool.h"
#include "../services/storage_request.h"
#include "basic_grpc.h"
#include "services/witness_bucket.h"
using namespace std;
using namespace ::dg::model;
namespace dg {
class SpringGrpcClientImpl final: public BasicGrpcClient < StorageRequest > {
public:
    SpringGrpcClientImpl(Config config,
                         MessagePool<StorageRequest> *storage_pool)
        : BasicGrpcClient<StorageRequest>(config,storage_pool), config_(&config) {

    }



    virtual void Run() {
        message_pool_->Run();
        test();
    }
    void test(){
        while(1){
        CallData data;
        data.func = [&data]() -> MatrixError {
          return (bind(&StorageRequest::storage,
                       ( StorageRequest *) data.apps)());
        };

        message_pool_->enqueue(&data);
        MatrixError error = data.Wait();
        }
    }

private:

    Config *config_;
};
}
#endif //MATRIX_APPS_SPRING_GRPC_H
