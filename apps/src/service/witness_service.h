/*
 * deepv_service.h
 *
 *  Created on: Apr 20, 2016
 *      Author: chenzhen
 */

#ifndef DEEPV_SERVICE_H_
#define DEEPV_SERVICE_H_

#include <iostream>
#include "model/proto/witness.grpc.pb.h"

using namespace std;

namespace dg {

class WitnessServiceImpl : public WitnessService::Service {
 public:

    grpc::Status Recognize(::grpc::ServerContext* context,
                           const ::dg::RecognizeRequest* request,
                           ::dg::RecognizeResponse* response) {
        cout << "Get Recognize request: " << request->sessionid()
             << ", Image URI:" << request->image().uri() << endl;
        response->mutable_result()->mutable_brand()->set_brandid(123);
        response->mutable_status()->set_msg("finish");
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
