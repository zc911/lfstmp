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
class SpringGrpcClientImpl {
public:
    SpringGrpcClientImpl(Config config)
        : config_(&config) {

    }

    virtual void Run() {
        StorageRequest sr(config_);
        while (1) {
            sr.storage();
        }
    }

private:

    Config *config_;
};
}
#endif //MATRIX_APPS_SPRING_GRPC_H
