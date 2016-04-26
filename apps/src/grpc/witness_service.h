/*
 * deepv_service.h
 *
 *  Created on: Apr 20, 2016
 *      Author: chenzhen
 */

#ifndef DEEPV_SERVICE_H_
#define DEEPV_SERVICE_H_

#include <unistd.h>
#include <iostream>
#include <grpc++/grpc++.h>
#include "model/witness.grpc.pb.h"
#include "basic_service.h"
#include "config/config.h"

using namespace std;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

namespace dg {

class WitnessServiceImpl : public WitnessService::Service 
{
 public:

    WitnessServiceImpl(Config *config)
            : config_(config) 
    {
    }

 private:
    Config *config_;

    grpc::Status Recognize(::grpc::ServerContext* context,
                           const ::dg::RecognizeRequest* request,
                           ::dg::RecognizeResponse* response) {
        cout << "Get Recognize request: " << request->sessionid()
             << ", Image URI:" << request->image().uri() << endl;
        cout << "Start processing: " << request->sessionid() << "..." << endl;
        sleep(5);
        response->mutable_result()->mutable_brand()->set_brandid(123);
        response->mutable_status()->set_msg("finish");
        cout << "Finish processing: " << request->sessionid() << "..." << endl;
        cout << "=======" << endl;
        return grpc::Status::OK;
    }

    grpc::Status BatchRecognize(::grpc::ServerContext* context,
                                const ::dg::BatchRecognizeRequest* request,
                                ::dg::BatchRecognizeResponse* response) {
        return grpc::Status::OK;
    }
};
}
#endif /* DEEPV_SERVICE_H_ */
