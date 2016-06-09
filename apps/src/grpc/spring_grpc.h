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
using grpc::ClientAsyncResponseReader;
using grpc::Status;
using grpc::CompletionQueue;
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
        AsyncClientCall *call = new AsyncClientCall;
        call->response_reader=stub_->AsyncIndex(&call->context,request,&cq_);
        call->response_reader->Finish(&call->reply,&call->status,(void *)call);
    }
    void AsyncCompleteRpc(){
        void *got_tag;
        bool ok=false;
        while(cq_.Next(&got_tag,&ok)){
            AsyncClientCall *call = static_cast<AsyncClientCall*>(got_tag);
            GPR_ASSERT(ok);
            if(call->status.ok())
                std::cout<<"receivced"<<std::endl;
            else
                std::cout<<"rpc failed"<<std::endl;
            delete call;
        }
    }
private:
    struct AsyncClientCall{
        NullMessage reply;
        ClientContext context;
        Status status;
        std::unique_ptr<ClientAsyncResponseReader<NullMessage> > response_reader;
    };
    std::unique_ptr<SpringService::Stub> stub_;
    CompletionQueue cq_;
};
#endif //MATRIX_APPS_SPRING_GRPC_H
