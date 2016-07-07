//
// Created by chenzhen on 6/1/16.
//

#ifndef PROJECT_ENGINE_SERVICE_H
#define PROJECT_ENGINE_SERVICE_H
#include "matrix_engine/model/model.h"
#include "model/witness.grpc.pb.h"
#include "engine_service.h"

using namespace ::dg::model;
namespace dg {

class EngineService {
public:
    virtual MatrixError Recognize(const WitnessRequest *request, WitnessResponse *response) { }

    virtual MatrixError BatchRecognize(const WitnessBatchRequest *request,
                                       WitnessBatchResponse *response) { }
    virtual ~EngineService() { }
};

}

#endif //PROJECT_ENGINE_SERVICE_H
