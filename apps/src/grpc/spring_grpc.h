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
    ClientContext context;
        CompletionQueue cq;
        Status status;
        std::unique_ptr<ClientAsyncResponseReader<NullMessage> > rpc(
            stub_->AsyncIndex(&context,request,&cq));
        rpc->Finish(reply,&status,(void*)1);
        void *got_tag;
        bool ok=false;
        cq.Next(&got_tag,&ok);
        if(status.ok()){
            return "grpc success";
        }else{
            return "grpc failed";
        }

    }
    void AsyncCompleteRpc(){
        void *got_tag;
        bool ok=false;
        while(cq_.Next(&got_tag,&ok)){
            AsyncClientCall *call = static_cast<AsyncClientCall*>(got_tag);
          //  GPR_ASSERT(ok);
            cout<<"Weg"<<endl;
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
