// Generated by the gRPC protobuf plugin.
// If you make any local change, they will be lost.
// source: witness.proto

#include "witness.pb.h"
#include "witness.grpc.pb.h"

#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/channel_interface.h>
#include <grpc++/impl/codegen/client_unary_call.h>
#include <grpc++/impl/codegen/method_handler_impl.h>
#include <grpc++/impl/codegen/rpc_service_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/sync_stream.h>
namespace dg {

static const char* WitnessService_method_names[] = {
  "/dg.WitnessService/Recognize",
  "/dg.WitnessService/BatchRecognize",
};

std::unique_ptr< WitnessService::Stub> WitnessService::NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options) {
  std::unique_ptr< WitnessService::Stub> stub(new WitnessService::Stub(channel));
  return stub;
}

WitnessService::Stub::Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel)
  : channel_(channel), rpcmethod_Recognize_(WitnessService_method_names[0], ::grpc::RpcMethod::NORMAL_RPC, channel)
  , rpcmethod_BatchRecognize_(WitnessService_method_names[1], ::grpc::RpcMethod::NORMAL_RPC, channel)
  {}

::grpc::Status WitnessService::Stub::Recognize(::grpc::ClientContext* context, const ::dg::WitnessRequest& request, ::dg::WitnessResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_Recognize_, context, request, response);
}

::grpc::ClientAsyncResponseReader< ::dg::WitnessResponse>* WitnessService::Stub::AsyncRecognizeRaw(::grpc::ClientContext* context, const ::dg::WitnessRequest& request, ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader< ::dg::WitnessResponse>(channel_.get(), cq, rpcmethod_Recognize_, context, request);
}

::grpc::Status WitnessService::Stub::BatchRecognize(::grpc::ClientContext* context, const ::dg::WitnessBatchRequest& request, ::dg::WitnessBatchResponse* response) {
  return ::grpc::BlockingUnaryCall(channel_.get(), rpcmethod_BatchRecognize_, context, request, response);
}

::grpc::ClientAsyncResponseReader< ::dg::WitnessBatchResponse>* WitnessService::Stub::AsyncBatchRecognizeRaw(::grpc::ClientContext* context, const ::dg::WitnessBatchRequest& request, ::grpc::CompletionQueue* cq) {
  return new ::grpc::ClientAsyncResponseReader< ::dg::WitnessBatchResponse>(channel_.get(), cq, rpcmethod_BatchRecognize_, context, request);
}

WitnessService::Service::Service() {
  (void)WitnessService_method_names;
  AddMethod(new ::grpc::RpcServiceMethod(
      WitnessService_method_names[0],
      ::grpc::RpcMethod::NORMAL_RPC,
      new ::grpc::RpcMethodHandler< WitnessService::Service, ::dg::WitnessRequest, ::dg::WitnessResponse>(
          std::mem_fn(&WitnessService::Service::Recognize), this)));
  AddMethod(new ::grpc::RpcServiceMethod(
      WitnessService_method_names[1],
      ::grpc::RpcMethod::NORMAL_RPC,
      new ::grpc::RpcMethodHandler< WitnessService::Service, ::dg::WitnessBatchRequest, ::dg::WitnessBatchResponse>(
          std::mem_fn(&WitnessService::Service::BatchRecognize), this)));
}

WitnessService::Service::~Service() {
}

::grpc::Status WitnessService::Service::Recognize(::grpc::ServerContext* context, const ::dg::WitnessRequest* request, ::dg::WitnessResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status WitnessService::Service::BatchRecognize(::grpc::ServerContext* context, const ::dg::WitnessBatchRequest* request, ::dg::WitnessBatchResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}


}  // namespace dg
