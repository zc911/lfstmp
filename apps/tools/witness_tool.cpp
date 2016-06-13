#include <time.h>
#include <thread>
#include <iostream>
#include <memory>
#include <vector>
#include <sys/time.h>
#include <grpc++/grpc++.h>
#include "model/witness.grpc.pb.h"
#include "pbjson/pbjson.hpp"
#include "codec/base64.h"
#include "string_util.h"

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
//    ctx->mutable_functions()->Add(8);
}

static void Print(const WitnessBatchResponse &resp) {
    cout << "=================" << endl;
    cout << "SessionId:" << resp.context().sessionid() << endl;
    for (int i = 0; i < resp.results().size(); ++i) {
        const WitnessResult &r = resp.results().Get(i);
        rapidjson::Value *value = pbjson::pb2jsonobject(&r);
        string s;
        pbjson::json2string(value, s);
        cout << s << endl;
    }
}

static void Print(const WitnessResponse &resp) {
    cout << "=================" << endl;
    cout << "SessionId:" << resp.context().sessionid() << endl;
    const WitnessResult &r = resp.result();
    cout<<r.vehicles_size()<<" vehicle size"<<endl;
    for(int i=0;i<r.vehicles_size();i++)
    cout<<r.vehicles(i).plate().platetext()<<endl;
    const RecVehicle &rv = r.vehicles(0);
    const WitnessResponseContext  &req=resp.context();
    rapidjson::Value *value = pbjson::pb2jsonobject(&req);
    string s;
    pbjson::json2string(value, s);
    cout << s << endl;
}
static void Print(const WitnessRequest &req) {
    cout << "=================" << endl;
    cout << "SessionId:" << req.context().sessionid() << endl;
    rapidjson::Value *value = pbjson::pb2jsonobject(&req);
    string s;
    pbjson::json2string(value, s);
    cout << s << endl;
}
static void PrintCost(string s, struct timeval &start, struct timeval &end) {
    cout << s << (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000 << endl;
}

class WitnessClient {
public:
    WitnessClient(std::shared_ptr<Channel> channel)
        : stub_(WitnessService::NewStub(channel)) {
    }

    void Recognize(const string file_path, const string session_id, bool uri = true) {
        WitnessRequest req;
        WitnessRequestContext *ctx = req.mutable_context();

        ctx->set_sessionid(session_id);
        WitnessImage *witnessimage =req.mutable_image();
        SetFunctions(ctx);
        if (uri){
            witnessimage->mutable_data()->set_uri(file_path);

        }else {
            string s = encode2base64(file_path.c_str());
            witnessimage->mutable_data()->set_bindata(s);
        }
        WitnessRelativeROI * roi = witnessimage->add_relativeroi();
        roi->set_posx(0);
        roi->set_posy(0);
        roi->set_width(1000);
        roi->set_height(1000);
        WitnessResponse resp;
        Print(req);
        ClientContext context;
        struct timeval start, end;
        gettimeofday(&start, NULL);
        Status status = stub_->Recognize(&context, req, &resp);
        gettimeofday(&end, NULL);

        if (status.ok()) {
            cout << "Rec finished: " << resp.context().sessionid() << endl;
            Print(resp);

            PrintCost("Rec cost:", start, end);
        } else {
            cout << "Rec error: " << status.error_message() << endl;
        }

    }

    void RecognizeBatch(vector<string> &file_paths, const string session_id, bool uri = true) {
        WitnessBatchRequest req;
        WitnessRequestContext *ctx = req.mutable_context();
        ctx->set_sessionid(session_id);
        SetFunctions(ctx);
        for (vector<string>::iterator itr = file_paths.begin(); itr != file_paths.end(); ++itr) {
            WitnessImage *image = req.add_images();

            if (uri){
                image->mutable_data()->set_uri(*itr);

            }
            else {
                string s = encode2base64((*itr).c_str());
                image->mutable_data()->set_bindata(s);
            }
            WitnessRelativeROI * roi = image->add_relativeroi();
            roi->set_posx(0);
            roi->set_posy(0);
            roi->set_width(100);
            roi->set_height(100);

        }

        WitnessBatchResponse resp;
        ClientContext context;
        struct timeval start, end;
        gettimeofday(&start, NULL);
        Status status = stub_->BatchRecognize(&context, req, &resp);
        gettimeofday(&end, NULL);
        if (status.ok()) {
            cout << "Batch Rec finished: " << resp.context().sessionid() << endl;
            Print(resp);
            PrintCost("Batch rec cost: ", start, end);
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
            cout << "Rec(Asyn) finished: " << reply.context().sessionid() << endl;
            cout << pbjson::pb2jsonobject(&reply)->GetString() << endl;
        } else {
            cout << "Rec(Asyn) error: " << status.error_message() << endl;
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
            cout << "Batch Rec(Asyn) finished: " << reply.context().sessionid() << endl;
            Print(reply);
        } else {
            cout << "Batch Rec(Asyn) error: " << status.error_message() << endl;
        }

    }

private:
    // Out of the passed in Channel comes the stub, stored here, our view of the
    // server's exposed services.
    std::unique_ptr<WitnessService::Stub> stub_;
};

static string RandomSessionId() {
    srand((uint) time(NULL));
    int id = rand();
    for (int i = 0; i < 10; ++i) {
        id = (id << 7) | rand();
    }

    id = id < 0 ? -1 * id : id;
    return std::to_string(id);
}

void callA(string address, string image_file_path, bool batch) {
    WitnessClientAsyn client(
        grpc::CreateChannel(string(address),
                            grpc::InsecureChannelCredentials()));
    while (1) {
        string id = RandomSessionId();
        if (batch) {
            cout << "Batch Rec asyn: " << id << endl;
            vector<string> images;
            images.push_back(image_file_path);
            images.push_back(image_file_path);
            images.push_back(image_file_path);
            images.push_back(image_file_path);
            images.push_back(image_file_path);
            images.push_back(image_file_path);
            images.push_back(image_file_path);
            images.push_back(image_file_path);
            client.RecognizeBatch(images, id);

        } else {
            cout << "Rec asyn: " << id << endl;
            client.Recognize(image_file_path, id);
        }
    }
}

void callS(string address, string image_file_path, bool batch, bool uri) {
    WitnessClient client(
        grpc::CreateChannel(string(address),
                            grpc::InsecureChannelCredentials()));
    while (1) {
        string id = RandomSessionId();
        if (batch) {
            cout << "Batch Rec syn: " << id << endl;
            vector<string> images;
            images.push_back(image_file_path);
            images.push_back(image_file_path);
            images.push_back(image_file_path);
            images.push_back(image_file_path);
            images.push_back(image_file_path);
            images.push_back(image_file_path);
            images.push_back(image_file_path);
            images.push_back(image_file_path);
            client.RecognizeBatch(images, id, uri);
        } else {
            cout << "Rec syn: " << id << endl;
            client.Recognize(image_file_path, id, uri);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        cout << "Usage: " << argv[0] << " IMAGE_FILE_PATH [S|A] [S|B] IP:PORT THREAD_NUM"
            << endl;
        return 0;
    }

    string image_file_path = string(argv[1]);

    char address[1024];
    sprintf(address, "%s", argv[4]);


    bool asyn = false;
    if (string(argv[2]) == "A") {
        asyn = true;
    }

    bool batch = false;
    if (string(argv[3]) == "B") {
        batch = true;
    }

    int threadNum = 1;
    threadNum = atoi(argv[5]);

    if (asyn) {
        callA(address, image_file_path, batch);

    } else {
        thread t(callS, address, image_file_path, batch, true);
        t.join();
    }

    cout << "Wait..." << endl;
    while (1) {
        std::this_thread::sleep_for(std::chrono::minutes(100000));
    }

}

