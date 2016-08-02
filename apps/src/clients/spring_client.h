#ifndef SRC_CLIENTS_SPRING_CLIENT_H_
#define SRC_CLIENTS_SPRING_CLIENT_H_
#include <google/protobuf/text_format.h>

#include "spring.grpc.pb.h"
#include "localcommon.pb.h"
#include "witness.grpc.pb.h"
#include "pbjson/pbjson.hpp"

#include "services/witness_bucket.h"

using ::dg::model::SpringService;
using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::Status;
using grpc::CompletionQueue;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
namespace dg{
	static int timeout = 5;

class SpringClient {
public:
    SpringClient(){

    }
	MatrixError IndexVehicle(string storageAddress,const VehicleObj &v){
        map<string, std::unique_ptr<SpringService::Stub> >::iterator it = stubs_.find(storageAddress);
        if (it == stubs_.end()) {
            CreateConnect(storageAddress);
        }
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
                MatrixError err;

        if (status.ok()) {
            VLOG(VLOG_SERVICE) << "send to storage success" << endl;

            return err;
        } else {
            VLOG(VLOG_SERVICE) << "send to storage failed " << status.error_code() << endl;
            stubs_.erase(stubs_.find(storageAddress));
            return err;
        }
	}
    void CreateConnect(string storageAddress) {
        shared_ptr<grpc::Channel> channel = grpc::CreateChannel(storageAddress, grpc::InsecureChannelCredentials());
        std::unique_ptr<SpringService::Stub> stub(SpringService::NewStub(channel));
        stubs_.insert(std::make_pair(storageAddress, std::move(stub)));
        if (stubs_.size() > 10) {
            stubs_.erase(stubs_.begin());
        }
        for (map<string, std::unique_ptr<SpringService::Stub> >::iterator it = stubs_.begin(); it != stubs_.end();
             it++) {
            VLOG(VLOG_SERVICE) << it->first;
        }

    };
private:
    map<string, std::unique_ptr<SpringService::Stub> > stubs_;

};
}

#endif
