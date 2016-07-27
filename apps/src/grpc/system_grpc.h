//
// Created by jiajaichen on 16-6-14.
//

#ifndef PROJECT_SYSTEM_GRPC_H_H
#define PROJECT_SYSTEM_GRPC_H_H

#include <grpc++/grpc++.h>
#include "services/system_service.h"

namespace dg {

class GrpcSystemServiceImpl final: public SystemService::Service {
public:
  GrpcSystemServiceImpl(Config config, string addr):config_(&config),addr_(addr),service_system_(&config, "system_service"){ 
  }
  virtual ~GrpcSystemServiceImpl() {
  }
      void Run() {

        grpc::ServerBuilder builder;
        builder.AddListeningPort(addr_, grpc::InsecureServerCredentials());
        builder.RegisterService(this);
        unique_ptr<grpc::Server> server(builder.BuildAndStart());
        cout  << "System Server(GRPC) listening on " << (int) config_->Value("System/Port")
             << endl;
        server->Wait();
    }

private:
  SystemAppsService service_system_;
  Config *config_;
  string addr_;
  virtual grpc::Status Ping(grpc::ServerContext* context, const PingRequest *request, PingResponse *response) override
  {
    struct timeval start, finish;
    gettimeofday(&start, NULL);

    MatrixError error = service_system_.Ping(request, response);
    gettimeofday(&finish, NULL);
    //  rapidjson::Value *value = pbjson::pb2jsonobject(response);
    //  string s;
    //   pbjson::json2string(value, s);

    return error.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;

  }


  virtual grpc::Status SystemStatus(grpc::ServerContext* context, const SystemStatusRequest *request, SystemStatusResponse *response) override
  {
    struct timeval start, finish;
    gettimeofday(&start, NULL);
    MatrixError error = service_system_.SystemStatus(request, response);
    gettimeofday(&finish, NULL);
    //  rapidjson::Value *value = pbjson::pb2jsonobject(response);
    //  string s;
    //   pbjson::json2string(value, s);

    return error.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;
  }

  /*    virtual grpc::Status GetInstances(grpc::ServerContext* context, const GetInstancesRequest *request, InstanceConfigureResponse *response) override
      {
          return service_.GetInstances(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
      }

      virtual grpc::Status ConfigEngine(grpc::ServerContext* context, const InstanceConfigureRequest *request, InstanceConfigureResponse *response) override
      {
          return service_.ConfigEngine(request, response) ? grpc::Status::OK : grpc::Status::CANCELLED;
      }*/
};

}
#endif //PROJECT_SYSTEM_GRPC_H_H
