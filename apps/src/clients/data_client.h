#ifndef SRC_CLIENTS_DATA_CLIENT_H_
#define SRC_CLIENTS_DATA_CLIENT_H_
#include "dataservice.grpc.pb.h"
#include "localcommon.pb.h"
#include "witness.grpc.pb.h"
#include "pbjson/pbjson.hpp"
#include <google/protobuf/text_format.h>
using ::model::DataService;
using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::Status;
using grpc::CompletionQueue;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
namespace dg{


class DataClient {
public:
    DataClient() {

    }
	MatrixError SendBatchData(string address,const VehicleObj &v){
        model::Vehicle pbVehicle;
        



        map<string, std::unique_ptr<DataService::Stub> >::iterator it = stubs_.find(address);
        if (it == stubs_.end()) {
            CreateConnect(address);
        }
        NullMessage reply;
        ClientContext context;
        std::chrono::system_clock::time_point
            deadline = std::chrono::system_clock::now() + std::chrono::seconds(timeout);
        context.set_deadline(deadline);
        CompletionQueue cq;
        Status status;
        std::unique_ptr<ClientAsyncResponseReader<NullMessage> > rpc(
            stubs_[address]->AsyncIndexVehicle(&context, v, &cq));
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
    void CreateConnect(string address) {
        shared_ptr<grpc::Channel> channel = grpc::CreateChannel(storageAddress, grpc::InsecureChannelCredentials());
        std::unique_ptr<DataService::Stub> stub(DataService::NewStub(channel));
        stubs_.insert(std::make_pair(address, std::move(stub)));
        if (stubs_.size() > 10) {
            stubs_.erase(stubs_.begin());
        }
        for (map<string, std::unique_ptr<DataService::Stub> >::iterator it = stubs_.begin(); it != stubs_.end();
             it++) {
            VLOG(VLOG_SERVICE) << it->first;
        }

    };
private:
    map<string, std::unique_ptr<DataService::Stub> > stubs_;

};
}
#endif 
