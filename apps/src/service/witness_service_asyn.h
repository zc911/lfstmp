/*
 * witness_service_asyn.h
 *
 *  Created on: Apr 21, 2016
 *      Author: chenzhen
 */

#ifndef WITNESS_SERVICE_ASYN_H_
#define WITNESS_SERVICE_ASYN_H_

#include <unistd.h>
#include <pthread.h>
#include <memory>
#include <iostream>
#include <string>
#include <thread>

#include <grpc++/grpc++.h>

#include "model/proto/witness.grpc.pb.h"
#include "basic_service.h"
#include "config/config.h"

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;
using namespace std;

namespace dg {
class WitnessServiceAsynImpl : public BasicService {

 public:
    WitnessServiceAsynImpl(Config *config)
            : config_(config) {
        addr_ = (string) config_->Value("System/Ip") + ":"
                + (string) config_->Value("System/Port");
    }

    ~WitnessServiceAsynImpl() {
        server_->Shutdown();
        // Always shutdown the completion queue after the server.
        cq_->Shutdown();
    }

    // There is no shutdown handling in this code.
    void Run() {

        ServerBuilder builder;
        // Listen on the given address without any authentication mechanism.
        builder.AddListeningPort(addr_, grpc::InsecureServerCredentials());
        // Register "service_" as the instance through which we'll communicate with
        // clients. In this case it corresponds to an *asynchronous* service.
        builder.RegisterService(&service_);
        // Get hold of the completion queue used for the asynchronous communication
        // with the gRPC runtime.
        cq_ = builder.AddCompletionQueue();
        // Finally assemble the server.
        server_ = builder.BuildAndStart();

        // Proceed to the server's main loop.

        pthread_t tid_, tid2_;
        typedef void* (*FUNC)(void*);
        FUNC callback = (FUNC) &WitnessServiceAsynImpl::HandleRpcs;
        pthread_create(&tid_, NULL, callback, (void*) this);
        pthread_create(&tid2_, NULL, callback, (void*) this);

        std::cout << "Server(Asyn) listening on " << addr_ << std::endl;
        pthread_join(tid_, NULL);

        //HandleRpcs();
    }

 private:
    // Class encompasing the state and logic needed to serve a request.
    class CallData {
     public:
        // Take in the "service" instance (in this case representing an asynchronous
        // server) and the completion queue "cq" used for asynchronous communication
        // with the gRPC runtime.
        CallData(WitnessService::AsyncService* service,
                 ServerCompletionQueue* cq)
                : service_(service),
                  cq_(cq),
                  responder_(&ctx_),
                  status_(CREATE) {
            // Invoke the serving logic right away.
            Proceed();
        }

        void Proceed() {
            if (status_ == CREATE) {
                // Make this instance progress to the PROCESS state.
                status_ = PROCESS;

                // As part of the initial CREATE state, we *request* that the system
                // start processing SayHello requests. In this request, "this" acts are
                // the tag uniquely identifying the request (so that different CallData
                // instances can serve different requests concurrently), in this case
                // the memory address of this CallData instance.
                service_->RequestRecognize(&ctx_, &request_, &responder_, cq_,
                                           cq_, this);
            } else if (status_ == PROCESS) {
                // Spawn a new CallData instance to serve new clients while we process
                // the one for this CallData. The instance will deallocate itself as
                // part of its FINISH state.
                new CallData(service_, cq_);

                // The actual processing.
                cout << "Get Recognize request: " << request_.sessionid()
                     << ", Image URI:" << request_.image().uri() << endl;
                cout << "Start processing(Asyn): " << request_.sessionid()
                     << "..." << endl;
                sleep(5);
                reply_.mutable_result()->mutable_brand()->set_brandid(123);
                reply_.mutable_status()->set_msg("finish");
                cout << "Finish processing(Asyn): " << request_.sessionid()
                     << "..." << endl;
                cout << "=======" << endl;
                // And we are done! Let the gRPC runtime know we've finished, using the
                // memory address of this instance as the uniquely identifying tag for
                // the event.
                status_ = FINISH;
                responder_.Finish(reply_, Status::OK, this);
            } else {
                GPR_ASSERT(status_ == FINISH);
                // Once in the FINISH state, deallocate ourselves (CallData).
                delete this;
            }
        }

     private:
        // The means of communication with the gRPC runtime for an asynchronous
        // server.
        WitnessService::AsyncService* service_;
        // The producer-consumer queue where for asynchronous server notifications.
        ServerCompletionQueue* cq_;
        // Context for the rpc, allowing to tweak aspects of it such as the use
        // of compression, authentication, as well as to send metadata back to the
        // client.
        ServerContext ctx_;

        // What we get from the client.
        RecognizeRequest request_;
        // What we send back to the client.
        RecognizeResponse reply_;

        // The means to get back to the client.
        ServerAsyncResponseWriter<RecognizeResponse> responder_;

        // Let's implement a tiny state machine with the following states.
        enum CallStatus {
            CREATE,
            PROCESS,
            FINISH
        };

        CallStatus status_;  // The current serving state.
    };

    // This can be run in multiple threads if needed.
    void HandleRpcs() {
        // Spawn a new CallData instance to serve new clients.
        new CallData(&service_, cq_.get());
        void* tag;  // uniquely identifies a request.
        bool ok;
        while (true) {
            // Block waiting to read the next event from the completion queue. The
            // event is uniquely identified by its tag, which in this case is the
            // memory address of a CallData instance.
            cq_->Next(&tag, &ok);
            GPR_ASSERT(ok);
            static_cast<CallData*>(tag)->Proceed();
        }
    }

    Config *config_;
    string addr_;
    std::unique_ptr<ServerCompletionQueue> cq_;
    WitnessService::AsyncService service_;
    std::unique_ptr<Server> server_;
};
}

#endif /* WITNESS_SERVICE_ASYN_H_ */
