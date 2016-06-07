//
// Created by jiajaichen on 16-6-6.
//

#ifndef MATRIX_APPS_SPRING_GRPC_H
#define MATRIX_APPS_SPRING_GRPC_H

#include <grpc++/grpc++.h>
#include "../model/witness.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using namespace ::dg::model;
class Index{
public:
    Index(std::shared_ptr<Channel> channel):stub_()
};
#endif //MATRIX_APPS_SPRING_GRPC_H
