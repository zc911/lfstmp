//
// Created by chenzhen on 5/25/16.
//

#include "witness_grpc.h"
#include <sys/time.h>
#include "debug_util.h"

namespace dg {

GrpcWitnessServiceImpl::GrpcWitnessServiceImpl(const Config *config) : witness_apps_(config, "aaa") {
    addr_ = (string) config->Value("System/Ip") + ":"
        + (string) config->Value("System/Port");
}
GrpcWitnessServiceImpl::~GrpcWitnessServiceImpl() { }

void GrpcWitnessServiceImpl::Run() {
    grpc::ServerBuilder builder;
    builder.AddListeningPort(addr_, grpc::InsecureServerCredentials());
    builder.RegisterService(this);
    unique_ptr <grpc::Server> server(builder.BuildAndStart());
    server->Wait();
}

grpc::Status GrpcWitnessServiceImpl::Recognize(grpc::ServerContext *context,
                                               const WitnessRequest *request,
                                               WitnessResponse *response) {
    MatrixError err = witness_apps_.Recognize(request, response);
    return err.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;
}

grpc::Status GrpcWitnessServiceImpl::BatchRecognize(grpc::ServerContext *context,
                                                    const WitnessBatchRequest *request,
                                                    WitnessBatchResponse *response) {
    std::thread::id threadId = std::this_thread::get_id();
    MatrixError err = witness_apps_.BatchRecognize(request, response);
    return err.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;
}

GrpcWitnessServiceAsynImpl::GrpcWitnessServiceAsynImpl(Config *config) : config_(config) {

    addr_ = (string) config->Value("System/Ip") + ":"
        + (string) config->Value("System/Port");
}

GrpcWitnessServiceAsynImpl::~GrpcWitnessServiceAsynImpl() {
    server_->Shutdown();
    // Always shutdown the completion queue after the server.
    cq_->Shutdown();
}

// There is no shutdown handling in this code.
void GrpcWitnessServiceAsynImpl::Run() {

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
    thread *oneThread;
    int gpuNum = (int) config_->Value("System/GpuNum");

    if (gpuNum <= 0) {
        LOG(FATAL) << "Gpu number invalid, check the machine or config file" << endl;
        return;
    }
    int threadsPerGpu = (int) config_->Value("System/ThreadsPerGpu");
    if (threadsPerGpu <= 0) {
        LOG(FATAL) << "ThreadsPerGpu invalid, check the config file" << endl;
        return;
    }

    for (int i = 0; i < gpuNum; ++i) {
        int gpuId = i;
        config_->AddEntry("System/GpuId", AnyConversion(gpuId));

        for (int j = 0; j < threadsPerGpu; ++j) {
            cout << "Start a Witness service thread on GPU: " << (int) config_->Value("System/GpuId") << endl;
            WitnessAppsService *witness_apps =
                new WitnessAppsService(config_, "apps" + std::to_string(gpuId) + "_" + std::to_string(j));
            oneThread = new std::thread(&GrpcWitnessServiceAsynImpl::HandleRpcs, this, witness_apps);
        }
    }

    oneThread->join();

}

// Class encompasing the state and logic needed to serve a request.
// Take in the "service" instance (in this case representing an asynchronous
// server) and the completion queue "cq" used for asynchronous communication
// with the gRPC runtime.
GrpcWitnessServiceAsynImpl::CallData::CallData(WitnessService::AsyncService *service,
                                               WitnessAppsService *witness_apps,
                                               ServerCompletionQueue *cq, bool batchMode)
    : batch_mode_(batchMode), service_(service),
      cq_(cq),
      batch_responder_(&ctx_), responder_(&ctx_),
      status_(CREATE) {
    // Invoke the serving logic right away.
    Proceed(NULL);
}

void GrpcWitnessServiceAsynImpl::CallData::Proceed(WitnessAppsService *witness_apps) {
    if (status_ == CREATE) {
        // Make this instance progress to the PROCESS state.
        status_ = PROCESS;

        // As part of the initial CREATE state, we *request* that the system
        // start processing requests. In this request, "this" acts are
        // the tag uniquely identifying the request (so that different CallData
        // instances can serve different requests concurrently), in this case
        // the memory address of this CallData instance.
        if (batch_mode_)
            service_->RequestBatchRecognize(&ctx_, &batch_request_, &batch_responder_, cq_,
                                            cq_, this);
        else
            service_->RequestRecognize(&ctx_, &request_, &responder_, cq_,
                                       cq_, this);

        cout << "Change from CREATE to PROCESS:  " << this << "mode:" << batch_mode_ << endl;

    } else if (status_ == PROCESS) {
        // Spawn a new CallData instance to serve new clients while we process
        // the one for this CallData. The instance will deallocate itself as
        // part of its FINISH state.

        new CallData(service_, NULL, cq_, this->batch_mode_);


        // The actual processing.
        cout << "======================" << endl;
        struct timeval start, end;
        if (request_.has_context() || request_.has_image()) {
            cout << "Get Recognize request: " << request_.context().sessionid() << endl;
            cout << "Start processing(Asyn): " << request_.context().sessionid()
                << "..." << endl;


            gettimeofday(&start, NULL);
            if (witness_apps)
                witness_apps->Recognize(&request_, &reply_);
            gettimeofday(&end, NULL);
            cout << "Rec cost: " << TimeCostInMs(start, end) << endl;
            cout << "Finish processing(Asyn): " << request_.context().sessionid()
                << "..." << endl;

            // And we are done! Let the gRPC runtime know we've finished, using the
            // memory address of this instance as the uniquely identifying tag for
            // the event.
            status_ = FINISH;
            responder_.Finish(reply_, Status::OK, this);
        } else if (batch_request_.has_context() || batch_request_.images_size() > 0) {
            cout << "Get Batch Recognize request: " << batch_request_.context().sessionid() << endl;
            cout << "Start processing(Asyn): " << batch_request_.context().sessionid()
                << "..." << endl;

            gettimeofday(&start, NULL);
            if (witness_apps)
                witness_apps->BatchRecognize(&batch_request_, &batch_reply_);
            gettimeofday(&end, NULL);
            cout << "Batch rec cost: " << TimeCostInMs(start, end) << endl;
            cout << "Finish batch processing(Asyn): " << batch_request_.context().sessionid()
                << "..." << endl;

            // And we are done! Let the gRPC runtime know we've finished, using the
            // memory address of this instance as the uniquely identifying tag for
            // the event.
            status_ = FINISH;
            batch_responder_.Finish(batch_reply_, Status::OK, this);
        }else{
            cout << "Bad call data " << endl;
        }

    } else {

        if (status_ != FINISH) {
            cout << "Status error, should be FINISH but " << status_ << endl;
        }
        // Once in the FINISH state, deallocate ourselves (CallData).
        delete this;
    }
}


// This can be run in multiple threads if needed.
void GrpcWitnessServiceAsynImpl::HandleRpcs(WitnessAppsService *witness_apps_) {
    // Spawn a new CallData instance to serve new clients.
    new CallData(&service_, NULL, cq_.get(), true);
    new CallData(&service_, NULL, cq_.get(), false);

    void *tag;  // uniquely identifies a request.
    bool ok;
    while (true) {
        // Block waiting to read the next event from the completion queue. The
        // event is uniquely identified by its tag, which in this case is the
        // memory address of a CallData instance.
        cq_->Next(&tag, &ok);
        cout << "Get call data from queue: "<< tag << endl;
        if (!ok) {
            cout << "Invalid tag:  " << tag << endl;
            continue;
        }
        static_cast<CallData *>(tag)->Proceed(witness_apps_);
    }
}

}

