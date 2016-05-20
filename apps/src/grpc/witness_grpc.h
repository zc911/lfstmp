/*============================================================================
 * File Name   : witness_grpc.h
 * Author      : yanlongtan@deepglint.com
 * Version     : 1.0.0.0
 * Copyright   : Copyright 2016 DeepGlint Inc.
 * Created on  : 04/15/2016
 * Description : 
 * ==========================================================================*/

#ifndef MATRIX_APPS_GRPC_WITNESS_H_
#define MATRIX_APPS_GRPC_WITNESS_H_

#include <thread>
#include <mutex>
#include <grpc++/grpc++.h>
#include "../model/common.pb.h"
#include "services/witness_service.h"

using namespace ::dg::model;
using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;

namespace dg {

class IGrpcWitnessService {
public:
    IGrpcWitnessService() {

    }
    virtual ~IGrpcWitnessService() { };
    virtual void Run() = 0;
};

class GrpcWitnessServiceImpl final: public IGrpcWitnessService, public WitnessService::Service {
public:
    GrpcWitnessServiceImpl(const Config *config) : witness_apps_(config) {
        addr_ = (string) config->Value("System/Ip") + ":"
            + (string) config->Value("System/Port");
    }
    virtual ~GrpcWitnessServiceImpl() { }
    void Run() {
        grpc::ServerBuilder builder;
        builder.AddListeningPort(addr_, grpc::InsecureServerCredentials());
        builder.RegisterService(this);
        unique_ptr<grpc::Server> server(builder.BuildAndStart());
        server->Wait();
    }
private:
    string addr_;
    WitnessAppsService witness_apps_;


    virtual grpc::Status Recognize(grpc::ServerContext *context,
                                   const WitnessRequest *request,
                                   WitnessResponse *response) override {
        MatrixError err = witness_apps_.Recognize(request, response);
        return err.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;
    }

    virtual grpc::Status BatchRecognize(grpc::ServerContext *context,
                                        const WitnessBatchRequest *request,
                                        WitnessBatchResponse *response) override {
        std::thread::id threadId = std::this_thread::get_id();
        cout << "Batch rec in thread id: " << hex << threadId << endl;
        MatrixError err = witness_apps_.BatchRecognize(request, response);
        return err.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;
    }
};

class GrpcWitnessServiceAsynImpl final: public IGrpcWitnessService, public WitnessService::Service {

public:
    GrpcWitnessServiceAsynImpl(const Config *config) {
        witness_apps_ = new WitnessAppsService(config);

        addr_ = (string) config->Value("System/Ip") + ":"
            + (string) config->Value("System/Port");
    }

    ~GrpcWitnessServiceAsynImpl() {
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

        HandleRpcs();
    }

private:
    // Class encompasing the state and logic needed to serve a request.
    class CallData {
    public:
        // Take in the "service" instance (in this case representing an asynchronous
        // server) and the completion queue "cq" used for asynchronous communication
        // with the gRPC runtime.
        CallData(WitnessService::AsyncService *service, WitnessAppsService *witness_apps,
                 ServerCompletionQueue *cq)
            : service_(service), witness_apps_(witness_apps),
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
                service_->RequestBatchRecognize(&ctx_, &request_, &responder_, cq_,
                                                cq_, this);

            } else if (status_ == PROCESS) {
                // Spawn a new CallData instance to serve new clients while we process
                // the one for this CallData. The instance will deallocate itself as
                // part of its FINISH state.
                new CallData(service_, witness_apps_, cq_);

                // The actual processing.
                cout << "Get Recognize request: " << request_.context().sessionid() << endl;
                cout << "Start processing(Asyn): " << request_.context().sessionid()
                    << "..." << endl;

                witness_apps_->BatchRecognize(&request_, &reply_);

                cout << "Finish processing(Asyn): " << request_.context().sessionid()
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
        WitnessService::AsyncService *service_;
        // The producer-consumer queue where for asynchronous server notifications.
        WitnessAppsService *witness_apps_;
        ServerCompletionQueue *cq_;
        // Context for the rpc, allowing to tweak aspects of it such as the use
        // of compression, authentication, as well as to send metadata back to the
        // client.
        ServerContext ctx_;

        // What we get from the client.
        WitnessBatchRequest request_;
        // What we send back to the client.
        WitnessBatchResponse reply_;

        // The means to get back to the client.
        ServerAsyncResponseWriter<WitnessBatchResponse> responder_;

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
        new CallData(&service_, witness_apps_, cq_.get());
        void *tag;  // uniquely identifies a request.
        bool ok;
        while (true) {
            // Block waiting to read the next event from the completion queue. The
            // event is uniquely identified by its tag, which in this case is the
            // memory address of a CallData instance.
            cq_->Next(&tag, &ok);
            GPR_ASSERT(ok);
            static_cast<CallData *>(tag)->Proceed();
        }
    }

//    Config *config_;
    string addr_;
    WitnessAppsService *witness_apps_;
    std::unique_ptr<ServerCompletionQueue> cq_;
    WitnessService::AsyncService service_;
    std::unique_ptr<Server> server_;
};

}

#endif //MATRIX_APPS_GRPC_WITNESS_H_
