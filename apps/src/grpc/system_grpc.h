//
// Created by jiajaichen on 16-6-14.
//

#ifndef PROJECT_SYSTEM_GRPC_H_H
#define PROJECT_SYSTEM_GRPC_H_H

#include <grpc++/grpc++.h>
#include "services/system_service.h"

namespace dg {

class GrpcSystemServiceImpl final: public BasicGrpcService<SystemAppsService>, public SystemService::Service {
public:
    GrpcSystemServiceImpl(Config config, string addr, MatrixEnginesPool <SystemAppsService> *engine_pool)
        : BasicGrpcService(config, addr, engine_pool) { }
    virtual ~GrpcSystemServiceImpl() { }
    virtual ::grpc::Service *service() {
        return this;
    };
private:

    grpc::Status Ping(grpc::ServerContext *context, const PingRequest *request, PingResponse *response) {
    /*    struct timeval start, finish;
        gettimeofday(&start, NULL);

        CallData data;
        data.func = [request, response, &data]() -> MatrixError {
          return (bind(&SystemAppsService::Ping,
                       (SystemAppsService *) data.apps,
                       placeholders::_1,
                       placeholders::_2))(request,
                                          response);
        };
        engine_pool_->enqueue(&data);
        MatrixError error = data.Wait();

        gettimeofday(&finish, NULL);
        //  rapidjson::Value *value = pbjson::pb2jsonobject(response);
        //  string s;
        //   pbjson::json2string(value, s);

        return error.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;*/
        return grpc::Status::OK;

    }
    grpc::Status SystemStatus(grpc::ServerContext *context,
                              const SystemStatusRequest *request,
                              SystemStatusResponse *response) {
     /*   struct timeval start, finish;
        gettimeofday(&start, NULL);

        CallData data;
        data.func = [request, response, &data]() -> MatrixError {
          return (bind(&SystemAppsService::SystemStatus,
                       (SystemAppsService *) data.apps,
                       placeholders::_1,
                       placeholders::_2))(request,
                                          response);
        };

        engine_pool_->enqueue(&data);
        MatrixError error = data.Wait();

        gettimeofday(&finish, NULL);
        //  rapidjson::Value *value = pbjson::pb2jsonobject(response);
        //  string s;
        //   pbjson::json2string(value, s);

        return error.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;*/
        return grpc::Status::OK;

    }
    /*   virtual grpc::Status Ping(grpc::ServerContext* context, const PingRequest *request, PingResponse *response) override
       {
           cout<<"hellos "<<endl;
           struct timeval start, finish;
           gettimeofday(&start, NULL);

           MatrixError error = service_system_.Ping(request,response);
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
           MatrixError error = service_system_.SystemStatus(request,response);
           gettimeofday(&finish, NULL);
           //  rapidjson::Value *value = pbjson::pb2jsonobject(response);
           //  string s;
           //   pbjson::json2string(value, s);

           return error.code() == 0 ? grpc::Status::OK : grpc::Status::CANCELLED;
       }

       /*  virtual grpc::Status GetInstances(grpc::ServerContext* context, const GetInstancesRequest *request, InstanceConfigureResponse *response) override
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
