//
// Created by jiajaichen on 16-6-6.
//

#ifndef MATRIX_APPS_SPRING_GRPC_H
#define MATRIX_APPS_SPRING_GRPC_H

#include <grpc++/grpc++.h>
#include "../model/witness.grpc.pb.h"
#include "../model/spring.grpc.pb.h"
using namespace std;
using namespace ::dg::model;
using ::dg::model::SpringService;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
class SpringGrpcClient{
public:
    SpringGrpcClient(std::shared_ptr<Channel> channel)
        : stub_(SpringService::NewStub(channel)) {
    }
    string Index(GenericObj &request,
                      NullMessage *reply){
        ClientContext context;
        Status status = stub_->Index(&context,request,reply);
        if (status.ok()) {
            return "reply successed";
        } else {
            return "reply failed";
        }
    }
private:
    std::unique_ptr<SpringService::Stub> stub_;
};
#endif //MATRIX_APPS_SPRING_GRPC_H
