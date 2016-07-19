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
#include "services/system_service.h"
#include "services/engine_pool.h"
#include "basic_grpc.h"

using namespace ::dg::model;
using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;

namespace dg {


class GrpcWitnessServiceImpl final: public BasicGrpcService<WitnessAppsService>, public WitnessService::Service {
public:
    GrpcWitnessServiceImpl(Config config, string addr, MatrixEnginesPool<WitnessAppsService> *engine_pool);
    virtual ~GrpcWitnessServiceImpl();
    virtual ::grpc::Service *service() {
        return this;
    };
private:
    virtual grpc::Status Recognize(grpc::ServerContext *context,
                                   const WitnessRequest *request,
                                   WitnessResponse *response);
    virtual grpc::Status Index(grpc::ServerContext *context,
                               const IndexRequest *request,
                               IndexResponse *response);
    virtual grpc::Status IndexTxt(grpc::ServerContext *context,
                                  const IndexTxtRequest *request,
                                  IndexTxtResponse *response);

    virtual grpc::Status BatchRecognize(grpc::ServerContext *context,
                                        const WitnessBatchRequest *request,
                                        WitnessBatchResponse *response);
    virtual void warmUp(int n) {
        string imgdata = ReadStringFromFile("warmup.dat", "rb");
        WitnessRequest protobufRequestMessage;
        WitnessResponse protobufResponseMessage;
        protobufRequestMessage.mutable_image()->mutable_data()->set_bindata(imgdata);
        protobufRequestMessage.mutable_image()->mutable_witnessmetadata()->set_sensorurl("http://127.0.0.1");
        WitnessRequestContext *ctx = protobufRequestMessage.mutable_context();
        ctx->mutable_functions()->Add(1);
        ctx->mutable_functions()->Add(2);
        ctx->mutable_functions()->Add(3);
        ctx->mutable_functions()->Add(4);
        ctx->mutable_functions()->Add(5);
        ctx->mutable_functions()->Add(6);
        ctx->mutable_functions()->Add(7);
        ctx->set_type(REC_TYPE_VEHICLE);

        for (int i = 0; i < n; i++) {
            CallData data;
            typedef MatrixError (*RecFunc)(WitnessAppsService *, const WitnessRequest *, WitnessResponse *);
            RecFunc rec_func = (RecFunc) &WitnessAppsService::Recognize;
            data.func = [rec_func, &protobufRequestMessage, &protobufResponseMessage, &data]() -> MatrixError {
              return (bind(rec_func, (WitnessAppsService *) data.apps,
                           placeholders::_1,
                           placeholders::_2))(&protobufRequestMessage,
                                              &protobufResponseMessage);
            };

            if (engine_pool_ == NULL) {
                LOG(ERROR) << "Engine pool not initailized. " << endl;
                return;
            }
            engine_pool_->enqueue(&data);

            MatrixError error = data.Wait();
        }

    }
};

}

#endif //MATRIX_APPS_GRPC_WITNESS_H_
