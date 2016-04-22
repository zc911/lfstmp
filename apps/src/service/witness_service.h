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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "model/proto/witness.grpc.pb.h"
#include "basic_service.h"
#include "config/config.h"

#include "model/frame.h"
#include "model/payload.h"
#include "engine/simple_engine.h"
#include "engine/witness_engine.h"

using namespace std;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

namespace dg {
namespace apps {
class WitnessServiceImpl : public apps::WitnessService::Service,
        public BasicService {
 public:

    WitnessServiceImpl(Config *config)
            : config_(config) {
        addr_ = (string) config_->Value("System/Ip") + ":"
                + (string) config_->Value("System/Port");
        engine_ = new WitnessEngine();

    }
    grpc::Status Recognize(::grpc::ServerContext* context,
                           const ::dg::apps::RecognizeRequest* request,
                           ::dg::apps::RecognizeResponse* response) {
        cout << "Get Recognize request: " << request->sessionid()
                << ", Image URI:" << request->image().uri() << endl;
        cout << "Start processing: " << request->sessionid() << "..." << endl;

        Frame *frame = new Frame(request->sessionid());
        DLOG(INFO) << "Read data from test.jpg" << endl;
        cv::Mat image = cv::imread("test.jpg");
        Payload *payload = new Payload(request->sessionid(), image);
        frame->set_payload(payload);

        engine_->Process(frame);

//        response->mutable_result()->mutable_brand()->set_brandid(123);
//        response->mutable_status()->set_msg("finish");
        cout << "Finish processing: " << request->sessionid() << "..." << endl;
        cout << "=======" << endl;
        return grpc::Status::OK;
    }

    grpc::Status BatchRecognize(
            ::grpc::ServerContext* context,
            const ::dg::apps::BatchRecognizeRequest* request,
            ::dg::apps::BatchRecognizeResponse* response) {
        return grpc::Status::OK;
    }

    void Run() {
        ServerBuilder builder;
        builder.AddListeningPort(addr_, grpc::InsecureServerCredentials());
        builder.RegisterService(this);
        std::unique_ptr<Server> server(builder.BuildAndStart());
        std::cout << "Server listening on " << addr_ << std::endl;
        server->Wait();
    }

 private:
    Config *config_;
    string addr_;
    WitnessEngine *engine_;
};
}
}
#endif /* DEEPV_SERVICE_H_ */
