//
// Created by jiajaichen on 16-6-15.
//

#ifndef PROJECT_STORAGE_REQUEST_H
#define PROJECT_STORAGE_REQUEST_H
#include "model/spring.grpc.pb.h"
#include "model/localcommon.pb.h"
#include "model/witness.grpc.pb.h"
#include "witness_bucket.h"
#include "pbjson/pbjson.hpp"
#include <google/protobuf/text_format.h>
using ::dg::model::SpringService;
using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::Status;
using grpc::CompletionQueue;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
namespace dg {
static int timeout = 5;
class StorageRequest {
public:
    StorageRequest(const Config *config) {
        string storageAddress = (string)config->Value(STORAGE_ADDRESS);
        createConnect(storageAddress);
    }

    MatrixError storage() {
        unique_lock<mutex> lock(WitnessBucket::Instance().mt_pop);
        VLOG(VLOG_SERVICE) << "========START REQUEST===========" << endl;

        MatrixError err;
        shared_ptr<WitnessVehicleObj> wv = WitnessBucket::Instance().Pop();
        string storageAddress = wv->storage().address();
  
        map<string, std::unique_ptr<SpringService::Stub> >::iterator it = stubs_.find(storageAddress);
        if (it == stubs_.end()){
            createConnect(storageAddress);
        }
        
        const VehicleObj &v = wv->vehicleresult();
        NullMessage reply;
        ClientContext context;
        std::chrono::system_clock::time_point
        deadline = std::chrono::system_clock::now() + std::chrono::seconds(timeout);
        context.set_deadline(deadline);
        CompletionQueue cq;
        Status status;
        std::unique_ptr<ClientAsyncResponseReader<NullMessage> > rpc(
            stubs_[storageAddress]->AsyncIndexVehicle(&context, v, &cq));
        rpc->Finish(&reply, &status, (void *) 1);
        void *got_tag;
        bool ok = false;
        cq.Next(&got_tag, &ok);
        if (status.ok()) {
            VLOG(VLOG_SERVICE) << "send to storage success" << endl;

            lock.unlock();
            return err;
        } else {
            VLOG(VLOG_SERVICE) << "send to storage failed " << status.error_code() << endl;
            stubs_.erase(stubs_.find(storageAddress));
            lock.unlock();
            return err;
        }
    }
    ~StorageRequest() { }
private:
    map<string, std::unique_ptr<SpringService::Stub> > stubs_;
    void createConnect(string storageAddress) {
        shared_ptr<grpc::Channel> channel = grpc::CreateChannel(storageAddress, grpc::InsecureChannelCredentials());
        std::unique_ptr<SpringService::Stub> stub(SpringService::NewStub(channel));
        stubs_.insert(std::make_pair(storageAddress, std::move(stub)));
        if(stubs_.size()>10){
            stubs_.erase( stubs_.begin() );
        }
        for(map<string, std::unique_ptr<SpringService::Stub> >::iterator it=stubs_.begin();it!=stubs_.end();it++){
            VLOG(VLOG_SERVICE)<<it->first;
        }

    };
};
}
#endif //PROJECT_STORAGE_REQUEST_H
