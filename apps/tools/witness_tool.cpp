#include <time.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <grpc++/grpc++.h>
#include "model/witness.grpc.pb.h"
#include "pbjson/pbjson.hpp"

using namespace std;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpc::ClientAsyncResponseReader;
using grpc::CompletionQueue;
using namespace dg;
using namespace dg::model;

static void SetFunctions(WitnessRequestContext *ctx) {
    ctx->mutable_functions()->Add(1);
    ctx->mutable_functions()->Add(2);
    ctx->mutable_functions()->Add(3);
    ctx->mutable_functions()->Add(4);
    ctx->mutable_functions()->Add(5);
    ctx->mutable_functions()->Add(6);
    ctx->mutable_functions()->Add(7);
    ctx->mutable_functions()->Add(8);
}

class WitnessClient {
public:
    WitnessClient(std::shared_ptr<Channel> channel)
        : stub_(WitnessService::NewStub(channel)) {
    }

    void Recognize(const string file_path, const string session_id) {
        WitnessRequest req;
        WitnessRequestContext *ctx = req.mutable_context();
        ctx->set_sessionid(session_id);
        SetFunctions(ctx);
        req.mutable_image()->mutable_data()->set_uri(file_path);

        WitnessResponse resp;
        ClientContext context;

        Status status = stub_->Recognize(&context, req, &resp);

        if (status.ok()) {
            cout << "Rec finished: " << resp.context().sessionid() << endl;
            cout << pbjson::pb2jsonobject(&resp)->GetString() << endl;
        } else {
            cout << "Rec error: " << status.error_message() << endl;
        }

    }

    void RecognizeBatch(vector<string> &file_paths, const string session_id) {
        WitnessBatchRequest req;
        WitnessRequestContext *ctx = req.mutable_context();
        ctx->set_sessionid(session_id);
        SetFunctions(ctx);
        for (vector<string>::iterator itr = file_paths.begin(); itr != file_paths.end(); ++itr) {
            req.add_images()->mutable_data()->set_uri(*itr);
        }

        WitnessBatchResponse resp;
        ClientContext context;

        Status status = stub_->BatchRecognize(&context, req, &resp);

        if (status.ok()) {
            cout << "Batch Rec finished: " << resp.context().sessionid() << endl;
            cout << pbjson::pb2jsonobject(&resp)->GetString() << endl;
        } else {
            cout << "Batch Rec error: " << status.error_message() << endl;
        }

    }

private:
    std::unique_ptr<WitnessService::Stub> stub_;
};

class WitnessClientAsyn {
public:
    explicit WitnessClientAsyn(std::shared_ptr<Channel> channel)
        : stub_(WitnessService::NewStub(channel)) {
    }

    // Assambles the client's payload, sends it and presents the response back
    // from the server.
    void Recognize(const string file_path, const string session_id) {
        // Data we are sending to the server.
        WitnessRequest request;
        WitnessRequestContext *ctx = request.mutable_context();
        ctx->set_sessionid(session_id);
        SetFunctions(ctx);

        request.mutable_image()->mutable_data()->set_uri(file_path);

        // Container for the data we expect from the server.
        WitnessResponse reply;

        // Context for the client. It could be used to convey extra information to
        // the server and/or tweak certain RPC behaviors.
        ClientContext context;

        // The producer-consumer queue we use to communicate asynchronously with the
        // gRPC runtime.
        CompletionQueue cq;

        // Storage for the status of the RPC upon completion.
        Status status;

        // stub_->AsyncSayHello() perform the RPC call, returning an instance we
        // store in "rpc". Because we are using the asynchronous API, we need the
        // hold on to the "rpc" instance in order to get updates on the ongoig RPC.
        cout << "Send request..." << endl;
        std::unique_ptr<ClientAsyncResponseReader<WitnessResponse> > rpc(
            stub_->AsyncRecognize(&context, request, &cq));

        // Request that, upon completion of the RPC, "reply" be updated with the
        // server's response; "status" with the indication of whether the operation
        // was successful. Tag the request with the integer 1.
        cout << "Wait for response..." << endl;
        rpc->Finish(&reply, &status, (void *) 1);
        void *got_tag;
        bool ok = false;
        // Block until the next result is available in the completion queue "cq".
        cq.Next(&got_tag, &ok);

        // Verify that the result from "cq" corresponds, by its tag, our previous
        // request.
        GPR_ASSERT(got_tag == (void *) 1);
        // ... and that the request was completed successfully. Note that "ok"
        // corresponds solely to the request for updates introduced by Finish().
        GPR_ASSERT(ok);

        // Act upon the status of the actual RPC.
        if (status.ok()) {
            cout << "Rec finished: " << reply.context().sessionid() << endl;
        } else {
            cout << "Rec error: " << status.error_message() << endl;
        }

    }

    void RecognizeBatch(vector<string> &file_paths, const string session_id) {
        // Data we are sending to the server.
        WitnessBatchRequest request;
        WitnessRequestContext *ctx = request.mutable_context();
        ctx->set_sessionid(session_id);
        SetFunctions(ctx);

        for (vector<string>::iterator itr = file_paths.begin(); itr != file_paths.end(); ++itr) {
            request.add_images()->mutable_data()->set_uri(*itr);
        }

        // Container for the data we expect from the server.
        WitnessBatchResponse reply;

        // Context for the client. It could be used to convey extra information to
        // the server and/or tweak certain RPC behaviors.
        ClientContext context;

        // The producer-consumer queue we use to communicate asynchronously with the
        // gRPC runtime.
        CompletionQueue cq;

        // Storage for the status of the RPC upon completion.
        Status status;

        // stub_->AsyncSayHello() perform the RPC call, returning an instance we
        // store in "rpc". Because we are using the asynchronous API, we need the
        // hold on to the "rpc" instance in order to get updates on the ongoig RPC.
        cout << "Send request..." << endl;
        std::unique_ptr<ClientAsyncResponseReader<WitnessBatchResponse> > rpc(
            stub_->AsyncBatchRecognize(&context, request, &cq));

        // Request that, upon completion of the RPC, "reply" be updated with the
        // server's response; "status" with the indication of whether the operation
        // was successful. Tag the request with the integer 1.
        cout << "Wait for response..." << endl;
        rpc->Finish(&reply, &status, (void *) 1);
        void *got_tag;
        bool ok = false;
        // Block until the next result is available in the completion queue "cq".
        cq.Next(&got_tag, &ok);

        // Verify that the result from "cq" corresponds, by its tag, our previous
        // request.
        GPR_ASSERT(got_tag == (void *) 1);
        // ... and that the request was completed successfully. Note that "ok"
        // corresponds solely to the request for updates introduced by Finish().
        GPR_ASSERT(ok);

        // Act upon the status of the actual RPC.
        if (status.ok()) {
            cout << "Rec finished: " << reply.context().sessionid() << endl;
        } else {
            cout << "Rec error: " << status.error_message() << endl;
        }

    }

private:
    // Out of the passed in Channel comes the stub, stored here, our view of the
    // server's exposed services.
    std::unique_ptr<WitnessService::Stub> stub_;
};

static string RandomSessionId() {
    srand(time(NULL));
    int id = rand();
    for (int i = 0; i < 10; ++i) {
        id = (id << 7) | rand();
    }

    id = id < 0 ? -1 * id : id;
    return std::to_string(id);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " IMAGE_FILE_PATH [S|A] IP:PORT"
            << endl;
        return 0;
    }

    string image_file_path = string(argv[1]);

    char address[1024];
    sprintf(address, "%s", argv[3]);

    bool asyn = false;
    if (string(argv[2]) == "A") {
        asyn = true;
    }

    vector<string> images;
    images.push_back(image_file_path);
    images.push_back(image_file_path);
    images.push_back(image_file_path);
    images.push_back(image_file_path);
    images.push_back(image_file_path);
    images.push_back(image_file_path);
    images.push_back(image_file_path);
    images.push_back(image_file_path);

    if (asyn) {
        WitnessClientAsyn client(
            grpc::CreateChannel(string(address),
                                grpc::InsecureChannelCredentials()));
        string id = RandomSessionId();
        cout << "Rec asyn: " << id << endl;
        client.RecognizeBatch(images, id);

    } else {
        WitnessClient client(
            grpc::CreateChannel(string(address),
                                grpc::InsecureChannelCredentials()));
        string id = RandomSessionId();
        cout << "Rec: " << id << endl;

        client.RecognizeBatch(images, id);
    }

}

